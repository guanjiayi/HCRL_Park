import argparse
import gym
import gym_hybrid
import numpy as np 
import torch
import wandb
import time
import os
import csv
from datetime import datetime
from easydict import EasyDict
from tqdm import tqdm 

from src.hcrl.hcrl_core import combined_shape
from src.hcrl.hcrl_core import discount_cumsum
from src.hcrl.hcrl_core import statistics_scalar
from src.hcrl.hcrl_core import HCRL
from src.hcrl.hcrl_model import HCRL_Model
from src.hcrl.hcrl_model_safe import HCRL_Model_Safe
from src.hcrl.hcrl_policy import HCRLPolicy
from src.hcrl.hcrl_policy_safe import HCRLPolicy_Safe


class HCRLBuffer_Safe:
    """
    A buffer for storing trajectories experienced by a HPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, discrete_act_dim, parameter_act_dim, size, device='cpu', gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)  
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)  
        self.discrete_act_buf = np.zeros(size, dtype=np.int64) 
        self.parameter_act_buf = np.zeros((size,parameter_act_dim), dtype=np.float32)
        # The sample for reward
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        # The sample for cost
        self.adv_cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ret_cost_buf = np.zeros(size, dtype=np.float32)
        self.val_cost_buf = np.zeros(size, dtype=np.float32)
        self.logp_discreate_act_buf = np.zeros(size, dtype=np.float32)
        self.logp_parameter_act_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.logit_action_type_buf = np.zeros((size,discrete_act_dim), dtype=np.float32)
        self.logit_action_argsmu_buf = np.zeros((size,parameter_act_dim), dtype=np.float32)
        self.logit_action_argssigma_buf = np.zeros((size,parameter_act_dim), dtype=np.float32)
        # self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device
        # define the return cost without the discount
        self.epret_cost_buf = np.zeros(size, dtype=np.float32)
        

    def store(self, obs, next_obs, discrete_act, parameter_act, 
              rew, val, logp_discrete_act, logp_parameter_act, done, 
              logit_action_type, logit_action_argsmu, logit_action_argssigma, 
              cost, val_cost):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.discrete_act_buf[self.ptr] = discrete_act
        self.parameter_act_buf[self.ptr] = parameter_act
        # store the sample for reward
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        # store the sample for cost
        self.cost_buf[self.ptr] = cost
        self.val_cost_buf[self.ptr] = val_cost
        self.logp_discreate_act_buf[self.ptr] = logp_discrete_act
        self.logp_parameter_act_buf[self.ptr] = logp_parameter_act
        self.done_buf[self.ptr] = done
        self.logit_action_type_buf[self.ptr] = logit_action_type
        self.logit_action_argsmu_buf[self.ptr] = logit_action_argsmu
        self.logit_action_argssigma_buf[self.ptr] = logit_action_argssigma
        self.ptr += 1

    def finish_path(self, last_val=0, last_val_cost = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_val_cost)
        vals_cost = np.append(self.val_cost_buf[path_slice], last_val_cost)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas_cost = costs[:-1] + self.gamma * vals_cost[1:] - vals_cost[:-1]
        self.adv_cost_buf[path_slice] = discount_cumsum(deltas_cost, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        # the next line computes cost-to-go, to be targets for the value function
        self.ret_cost_buf[path_slice] = discount_cumsum(costs, self.gamma)[:-1]

        # compute cost-to-go without the discount
        self.epret_cost_buf[path_slice] = discount_cumsum(costs, 1)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # the next two lines implement the advantage normalization trick
        adv_cost_mean, adv_cost_std = statistics_scalar(self.adv_cost_buf)
        self.adv_cost_buf = (self.adv_cost_buf - adv_cost_mean) / adv_cost_std

        data = dict(
            obs=self.obs_buf, 
            next_obs = self.next_obs_buf,
            discrete_act=self.discrete_act_buf,
            parameter_act=self.parameter_act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf, 
            ret_cost = self.ret_cost_buf,
            adv_cost = self.adv_cost_buf,
            logp_discrete_act = self.logp_discreate_act_buf,
            logp_parameter_act = self.logp_parameter_act_buf,
            done = self.done_buf,
            reward = self.rew_buf,
            cost = self.cost_buf,
            logit_action_type = self.logit_action_type_buf,
            logit_action_argsmu = self.logit_action_argsmu_buf,
            logit_action_argssigma = self.logit_action_argssigma_buf,
            epret_cost = self.epret_cost_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}

def record_result(run_name, env_id, eval_result, algorithm):
    '''
    Record the result of the evaluation.
    '''
    # set the directory and file name
    # env_dir = env_id+'_result'
    # result_dir = os.path.join(env_dir, run_name)
    result_dir = os.path.join('Result', algorithm, env_id, run_name)
    result_file = 'eval_result.csv'

    # Create the directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Full file path for the result of evaluation.
    result_eval_filepath = os.path.join(result_dir, result_file)
    
    # Write data to a CSV file
    with open(result_eval_filepath, 'w', newline='') as file:
        writer_result = csv.writer(file)
        writer_result.writerow([
            'eval_mean_ep_ret',
            'eval_mean_ep_cost',
            'eval_mean_ep_len',
            'eval_success_ratio'
        ])
        writer_result.writerows(eval_result)
    
    return True

def set_seed(seed):
    '''
    Set the seed for the torch and numpy
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)

def save_model(
        model, 
        optimizer, 
        epoch, 
        success_ratio, 
        mean_cost, 
        run_name,
        env_name='Perpendicular',
        algorithm = 'Hppo_safe',        
    ):
    """
    Save the model parameters, optimizer state, and the current epoch.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current training epoch.
        save_path (str): Path to save the model and optimizer state.
    """
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # current_time_dir = f"{current_time}"
    epoch_dir = f"{epoch}"
    mean_success_dir = f"{success_ratio}"
    mean_cost_dir = f"{mean_cost}"

    # model_path = 'Hppo_model_pth'
    # model_path_dir = os.path.join(model_path, env_name, run_name)
    model_path_dir = os.path.join('Model_path', algorithm, env_name, run_name)
    model_filename = 'model'+'_'+ epoch_dir+'_'+ mean_success_dir + '_'+ mean_cost_dir + '_' +'.pth'

    # Create the directory if it doesn't exist
    if not os.path.exists(model_path_dir):
        os.makedirs(model_path_dir)

     # Full file path for the observation trajctory
    save_model_directory = os.path.join(model_path_dir, model_filename)


    checkpoint = {
        'model_state_dict': model.state_dict(),
        'actor_optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, save_model_directory)
    print(f"Model saved to {save_model_directory}")


def load_model(model, optimizer, load_path, device='cpu'):
    """
    Load the model parameters, optimizer state, and training epoch.

    Args:
        model (nn.Module): The model to load the parameters into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        load_path (str): Path to the saved model checkpoint.
        device (str): Device to map the model and optimizer (default: 'cpu').

    Returns:
        int: The epoch number at which training can resume.
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from {load_path}")
    return True

    
def main(args):
    '''
    The main function for the train, evaluation
    '''
    if args.wandb:
        project_name = 'hybrid_parking_safe_parallel'
        run = wandb.init(project = project_name)
        run_name = wandb.run.name
    else:
        run_name = 'not_wandb'
    
    torch.set_num_threads(16)
    # Set the seed
    set_seed(args.seed)
    # Crete the environment
    env = gym.make(args.env)
    env.seed(args.seed)

    # Define the replay buffer
    buf = HCRLBuffer_Safe(
        obs_dim = args.state_dim,
        discrete_act_dim = args.discrete_action_dim,
        parameter_act_dim = args.parameter_action_dim,
        size = args.steps_per_epoch,
        device = args.device
    )

    # Define the HPPO Model 
    hppo_model = HCRL_Model_Safe(
        obs_shape = args.state_dim,
        discrete_act_dim = args.discrete_action_dim,
        parameter_act_dim = args.parameter_action_dim,
        # share_encoder = True,
        share_encoder = False,
        encoder_hidden_size_list = args.encoder_hidden_size_list,
        sigma_type = args.sigma_type,
        fixed_sigma_value = args.fixed_sigma_value,
        bound_type = args.bound_type,
    )

    # Define the HPPO Policy
    hppo_policy = HCRLPolicy_Safe(
        env_id = args.env,
        buf = buf,
        model = hppo_model,
        device = args.device,
        wandb_flag = args.wandb,
        env = env,
        # share_encoder=True,
        share_encoder = False,
        save_freq = 10,
        run_name = run_name,
    )

    best_performance = float('-inf') 
    eval_result = []

    # Train、Evualtion、collect for the HPPO policy and parking
    for epoch in tqdm(range(args.max_train_epochs), desc='Train Loop:'):

        # evaluation the hppo policy
        mean_info = hppo_policy.evaluate(
            eval_epoch = args.eval_epoch,
            algorithm = 'Hppo_safe',
            env_id = args.env,
            run_name = run_name,
        )

        mean_reward = mean_info['eval_mean_ep_ret']
        best_performance = max(best_performance, mean_reward)
        # record the result of the evaluation
        eval_result.append(
                (mean_info['eval_mean_ep_ret'],
                mean_info['eval_mean_ep_cost'],
                mean_info['eval_mean_ep_len'],
                mean_info['eval_success_ratio'])
            )

        # collect the sample and store to buffer
        hppo_policy.rollout(
            steps_per_epoch = args.steps_per_epoch
        )

        if (epoch % args.save_freq == 0) and (mean_reward >= 0.9*best_performance):
            save_model(
                hppo_policy._model,
                hppo_policy._actor_optimizer,
                epoch,
                mean_info['eval_success_ratio'], 
                mean_info['eval_mean_ep_cost'],
                run_name,
                algorithm = 'Hppo_safe'
            )
        
        # update the hppo policy
        hppo_policy.update(
            data = buf,
            train_iters=args.train_steps_per_epoch,
            epoch_step = epoch,
            best_performance = best_performance,
            current_reward = mean_reward,
            run_name = run_name,
            max_epoch_step = args.max_train_epochs,
            cost_limit = args.cost_limit,
            limit_delta = args.limit_delta,
        )

    record_flag = record_result(
        run_name = run_name,
        env_id = args.env,
        eval_result = eval_result,
        algorithm = 'Hppo_safe',
    )

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # set the parameter for the environment
    parser.add_argument('--env', type=str, default='Parallel_safe-v0')
    parser.add_argument('--state_dim', type=int, default=17, help='The dimensions of the observation')
    parser.add_argument('--discrete_action_dim', type=int, default=3, help='The dimensions of the discrete action')
    parser.add_argument('--parameter_action_dim', type=int, default=1, help='The dimensions of the parameters actions')
    # set the parameter for the training
    parser.add_argument('--steps_per_epoch', type=int, default=640, help='The epoch number of the buffer')
    parser.add_argument('--max_train_epochs', type=int, default=7000, help='The max epochs of the train')
    parser.add_argument('--train_steps_per_epoch', type=int, default=10, help='The train step of each epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size during train' )
    parser.add_argument('--random_epochs', type=int, default=3, help='get the sample by the random policy')
    # Set the parameter for the network
    parser.add_argument('--encoder_hidden_size_list', type=list, default=[256, 128, 64, 64], help='The hidden size for the encoder network')
    # parser.add_argument('--sigma_type', type=str, default='fixed')
    parser.add_argument('--sigma_type', type=str, default='fixed')
    # parser.add_argument('--sigma_type', type=str, default='constent')
    parser.add_argument('--fixed_sigma_value', type=float, default=0.3)
    parser.add_argument('--bound_type', type=str, default='tanh')
    # parser.add_argument('--bound_type', type=str, default='sigmoid')
    # Set the others parameter
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--device', default='cuda:0', help='Set the training device')
    # parser.add_argument('--device', default='cpu', help='Set the training device')
    parser.add_argument('--wandb','-wan',action='store_true',help='Flag for logging data via wandb')
    parser.add_argument('--cost_limit', type=float, default=10)
    parser.add_argument('--limit_delta', type=float, default=2)
    parser.add_argument('--save_freq', type=float, default=20)
    parser.add_argument('--model_dir', type=str, default='./hcrl_Perendicular/gentle-hill-34/model20241210_210237_1.0_8.504730606745541.pth' )

    args = parser.parse_args()

    main(args)
    print('success!')
