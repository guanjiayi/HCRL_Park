import torch
import time
from abc import ABC
from easydict import EasyDict
import numpy as np
from src.hcrl.hcrl_policy_utils import Adam
from collections import namedtuple
from typing import Optional, Tuple
# from torch.distributions import Independent, Normal, Categorical
from torch.distributions import Independent, Normal

from torch.distributions.categorical import Categorical

import os
import gym
import csv
import wandb
import gym_hybrid
from datetime import datetime
from src.hcrl.hcrl_model import RunningMeanStd

gae_data = namedtuple('gae_data', ['value', 'next_value', 'reward', 'done', 'traj_flag'])
ppo_policy_data = namedtuple('ppo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight'])
ppo_data = namedtuple(
    'ppo_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'adv', 'return_', 'weight']
)
ppo_policy_loss = namedtuple('ppo_policy_loss', ['policy_loss', 'entropy_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])
ppo_loss = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])


def gae(value, next_value, reward, done, gamma: float = 0.99, lambda_: float = 0.95) -> torch.FloatTensor:

    
    done = done.float()
    traj_flag = done
    if len(value.shape) == len(reward.shape) + 1:  # for some marl case: value(T, B, A), reward(T, B)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        traj_flag = traj_flag.unsqueeze(-1)

    next_value *= (1 - done)
    delta = reward + gamma * next_value - value
    factor = gamma * lambda_ * (1 - traj_flag)
    adv = torch.zeros_like(value)
    gae_item = torch.zeros_like(value[0])

    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item
    return adv

class HCRLPolicy(ABC):
    def __init__(
            self,
            env_id, 
            buf,
            model,
            device,
            ppo_param_init = True, 
            learning_rate: float = 3e-4,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            gamma = 0.99, 
            gae_lambda = 0.95, 
            recompute_adv = True,
            value_weight = 0.5,
            entropy_weight = 0.5,
            # clip_ratio = 0.05,
            clip_ratio = 0.2,
            adv_norm = True,
            value_norm = True,
            wandb_flag = False,
            env = None,
            share_encoder = True,
            batch_size = 64,
            save_freq = 10,
    )-> None:
        self._model = model.to(device)
        self._env_id = env_id
        self._buf = buf
        self._device = device
        self._ppo_param_init = ppo_param_init
        self._learning_rate = learning_rate
        self._grad_clip_type = grad_clip_type
        self._grad_clip_value = grad_clip_value
        
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._recompute_adv = recompute_adv
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight
        self._clip_ratio = clip_ratio
        self._adv_norm = adv_norm
        self._value_norm = value_norm
        self._wandb_flag = wandb_flag
        self._env = env
        self._share_encoder = share_encoder
        self._save_freq = save_freq
        self.batch_size = batch_size
        self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)


        # Init the model of the HPPO network
        if self._ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                if isinstance(m, torch.nn.Linear):
                    # orthogonal initialization
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)

            for m in self._model.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        if self._share_encoder:
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )
        else:
            self._actor_optimizer = Adam(
                self._model.actor.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )
            self._critic_optimizer = Adam(
                self._model.critic.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )


    def update(self, 
               data, 
               train_iters=10,
               epoch_step = 1,
               best_performance = float('-inf'), 
               current_reward = float('-inf'),
               run_name = 'not_wandb',
        )-> None:
        """
        Overview:
            Given training data, implement network update for one iteration and update related variables.
            Learner's API for serial entry.
            Also called in ``start`` for each iteration's training.
        Arguments:
            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.

        .. note::

            ``_policy`` must be set before calling this method.

            ``_policy.forward`` method contains: forward, backward, grad sync(if in multi-gpu mode) and
            parameter update.

            ``before_iter`` and ``after_iter`` hooks are called at the beginning and ending.
        """
        # save the current model
        if (epoch_step % self._save_freq == 0) and (current_reward >= 0.9*best_performance):   
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_time_dir = f"{current_time}"
            mean_reward_dir = f"{current_reward}"

            # Define the directory and file name
            if self._share_encoder:
                model_path = 'share_model_pth'
            else:
                model_path = 'model_pth'

            model_path_dir = os.path.join(model_path, run_name)
            model_filename = 'model'+current_time_dir+'_'+ mean_reward_dir+'.pth'


            # Create the directory if it doesn't exist
            if not os.path.exists(model_path_dir):
                os.makedirs(model_path_dir)

            # Full file path for the observation trajctory
            self._save_model_directory = os.path.join(model_path_dir, model_filename)
         
            # Save the model
            if self._share_encoder:
                model_state = {     
                    'model_state_dict': self._model.state_dict(),
                    'model_optimizer_state_dict': self._optimizer.state_dict(),
                }
            else:
                model_state = {     
                    'actor_state_dict': self._model.actor.state_dict(),
                    'critic_state_dict': self._model.critic.state_dict(),
                    'actor_optimizer_state_dict': self._actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': self._critic_optimizer.state_dict(),
                }

            torch.save(model_state, self._save_model_directory) 

        # train_iters = 10
        value_weight = 0.5
        # entropy_weight = 0.2
        entropy_weight = 0.5
        clip_ratio = 0.2
        
        batch_iters = 10

        # batch = data.get()
        batch = data.get()


        for epoch in range(train_iters):
            if self._share_encoder:
                with torch.no_grad():
                    value = self._model.compute_critic(batch['obs'])['value']
                    next_value = self._model.compute_critic(batch['next_obs'])['value']
                    #value norm
                    value *= self._running_mean_std.std
                    next_value *= self._running_mean_std.std
                    batch['adv']=gae(value,next_value,batch['reward'],batch['done'])
                    
                    unnormalized_returns = value + batch['adv']
                    
                    batch['value'] = value / self._running_mean_std.std
                    batch['ret'] = unnormalized_returns / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                
                for i in range(batch_iters):
                    batch_train = dict(
                        obs=batch['obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        next_obs=batch['next_obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        discrete_act=batch['discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        parameter_act=batch['parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        reward=batch['reward'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret=batch['ret'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv=batch['adv'][i*self.batch_size:(i+1)*self.batch_size,],
                        value=batch['value'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_discrete_act=batch['logp_discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_parameter_act=batch['logp_parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        done=batch['done'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_type=batch['logit_action_type'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argsmu=batch['logit_action_argsmu'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argssigma=batch['logit_action_argssigma'][i*self.batch_size:(i+1)*self.batch_size,],
                    )
                    output = self._model.compute_actor_critic(batch_train['obs'])
                    adv = batch_train['adv']
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    #discrete loss
                    discrete_weight = torch.ones_like(adv)
                    dist_discrete_new = Categorical(logits=output['logit']['action_type'])
                    dist_discrete_old = Categorical(logits=batch_train['logit_action_type'])
                    logp_discrete_new = dist_discrete_new.log_prob(batch_train['discrete_act'])
                    logp_discrete_old = dist_discrete_old.log_prob(batch_train['discrete_act'])
                    dist_discrete_new_entropy = dist_discrete_new.entropy()
                    if dist_discrete_new_entropy.shape != discrete_weight.shape:
                        dist_discrete_new_entropy = dist_discrete_new.entropy().mean(dim=1)
                    discrete_entropy_loss = (dist_discrete_new_entropy*discrete_weight).mean()
                    discrete_ratio = torch.exp(logp_discrete_new-logp_discrete_old)
                    if discrete_ratio.shape !=adv.shape:
                        discrete_ratio = discrete_ratio.mean(dim=1)
                    discrete_surr1 = discrete_ratio * adv
                    discrete_surr2 = discrete_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
                    discrete_policy_loss = (-torch.min(discrete_surr1, discrete_surr2) * discrete_weight).mean()
                    with torch.no_grad():
                        dis_approx_kl = (logp_discrete_old - logp_discrete_new).mean().item()
                        dis_clipped = discrete_ratio.gt(1 + clip_ratio) | discrete_ratio.lt(1 - clip_ratio)
                        dis_clipfrac = torch.as_tensor(dis_clipped).float().mean().item()
                        
                    #continuous loss
                    args_weight = torch.ones_like(adv)
                    dist_args_new = Independent(Normal(output['logit']['action_args']['mu'], output['logit']['action_args']['sigma']), 1)
                    dist_args_old = Independent(Normal(batch_train['logit_action_argsmu'], batch_train['logit_action_argssigma']), 1)
                    logp_args_new = dist_args_new.log_prob(batch_train['parameter_act'])
                    logp_args_old = dist_args_old.log_prob(batch_train['parameter_act'])
                    args_entropy_loss = (dist_args_new.entropy() * args_weight).mean()
                    args_ratio = torch.exp(logp_args_new - logp_args_old)
                    args_surr1 = args_ratio * adv
                    args_surr2 = args_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
                    args_policy_loss = (-torch.min(args_surr1, args_surr2) * args_weight).mean()
                    with torch.no_grad():
                        args_approx_kl = (logp_args_old - logp_args_new).mean().item()
                        args_clipped = args_ratio.gt(1 + clip_ratio) | args_ratio.lt(1 - clip_ratio)
                        args_clipfrac = torch.as_tensor(args_clipped).float().mean().item()
                        
                    #value loss
                    value_clip = batch_train['value'] + (output['value'] - batch_train['value']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret'] - output['value']).pow(2)
                    v2 = (batch_train['ret'] - value_clip).pow(2)
                    value_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()
                    
                    total_loss = discrete_policy_loss + args_policy_loss + value_weight*value_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss)
                    
                    self._optimizer.zero_grad()
                    total_loss.backward()
                    self._optimizer.step()
                    
                    if self._wandb_flag:
                        wandb.log({'record/discrete_policy_loss': discrete_policy_loss.item(), 
                                'record/args_policy_loss': args_policy_loss.item(), 
                                'record/value_loss':value_loss.item(),
                                'record/discrete_entropy_loss': discrete_entropy_loss.item(),
                                'record/args_entropy_loss': args_entropy_loss.item(),
                                'record/dis_approx_kl:': dis_approx_kl,
                                'record/dis_clipfrac:': dis_clipfrac,
                                'record/args_approx_kl:': args_approx_kl,
                                'record/args_clipfrac:': args_clipfrac,
                                })
                
            else:
                with torch.no_grad():
                    value = self._model.compute_critic(batch['obs'])['value']
                    next_value = self._model.compute_critic(batch['next_obs'])['value']
                    #value norm
                    value *= self._running_mean_std.std
                    next_value *= self._running_mean_std.std
                    batch['adv']=gae(value,next_value,batch['reward'],batch['done'])
                    
                    unnormalized_returns = value + batch['adv']
                    
                    batch['value'] = value / self._running_mean_std.std
                    batch['ret'] = unnormalized_returns / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                
                for i in range(batch_iters):
                    batch_train = dict(
                        obs=batch['obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        next_obs=batch['next_obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        discrete_act=batch['discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        parameter_act=batch['parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        reward=batch['reward'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret=batch['ret'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv=batch['adv'][i*self.batch_size:(i+1)*self.batch_size,],
                        value=batch['value'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_discrete_act=batch['logp_discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_parameter_act=batch['logp_parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        done=batch['done'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_type=batch['logit_action_type'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argsmu=batch['logit_action_argsmu'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argssigma=batch['logit_action_argssigma'][i*self.batch_size:(i+1)*self.batch_size,],
                    )
                    output = self._model.compute_actor_critic(batch_train['obs'])
                    adv = batch_train['adv']
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    #discrete loss
                    discrete_weight = torch.ones_like(adv)
                    dist_discrete_new = Categorical(logits=output['logit']['action_type'])
                    dist_discrete_old = Categorical(logits=batch_train['logit_action_type'])
                    logp_discrete_new = dist_discrete_new.log_prob(batch_train['discrete_act'])
                    logp_discrete_old = dist_discrete_old.log_prob(batch_train['discrete_act'])
                    dist_discrete_new_entropy = dist_discrete_new.entropy()
                    if dist_discrete_new_entropy.shape != discrete_weight.shape:
                        dist_discrete_new_entropy = dist_discrete_new.entropy().mean(dim=1)
                    discrete_entropy_loss = (dist_discrete_new_entropy*discrete_weight).mean()
                    discrete_ratio = torch.exp(logp_discrete_new-logp_discrete_old)
                    if discrete_ratio.shape !=adv.shape:
                        discrete_ratio = discrete_ratio.mean(dim=1)
                    discrete_surr1 = discrete_ratio * adv
                    discrete_surr2 = discrete_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
                    discrete_policy_loss = (-torch.min(discrete_surr1, discrete_surr2) * discrete_weight).mean()
                    with torch.no_grad():
                        dis_approx_kl = (logp_discrete_old - logp_discrete_new).mean().item()
                        dis_clipped = discrete_ratio.gt(1 + clip_ratio) | discrete_ratio.lt(1 - clip_ratio)
                        dis_clipfrac = torch.as_tensor(dis_clipped).float().mean().item()
                        
                    #continuous loss
                    args_weight = torch.ones_like(adv)
                    dist_args_new = Independent(Normal(output['logit']['action_args']['mu'], output['logit']['action_args']['sigma']), 1)
                    dist_args_old = Independent(Normal(batch_train['logit_action_argsmu'], batch_train['logit_action_argssigma']), 1)
                    logp_args_new = dist_args_new.log_prob(batch_train['parameter_act'])
                    logp_args_old = dist_args_old.log_prob(batch_train['parameter_act'])
                    args_entropy_loss = (dist_args_new.entropy() * args_weight).mean()
                    args_ratio = torch.exp(logp_args_new - logp_args_old)
                    args_surr1 = args_ratio * adv
                    args_surr2 = args_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
                    args_policy_loss = (-torch.min(args_surr1, args_surr2) * args_weight).mean()
                    with torch.no_grad():
                        args_approx_kl = (logp_args_old - logp_args_new).mean().item()
                        args_clipped = args_ratio.gt(1 + clip_ratio) | args_ratio.lt(1 - clip_ratio)
                        args_clipfrac = torch.as_tensor(args_clipped).float().mean().item()
                        
                    #value loss
                    value_clip = batch_train['value'] + (output['value'] - batch_train['value']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret'] - output['value']).pow(2)
                    v2 = (batch_train['ret'] - value_clip).pow(2)
                    value_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()
                    
                    # total_loss = discrete_policy_loss + args_policy_loss + value_weight*value_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss)

                    actor_loss = discrete_policy_loss + args_policy_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss)

                    critic_loss = value_loss

                    # self._optimizer.zero_grad()
                    # total_loss.backward()
                    # self._optimizer.step()
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self._actor_optimizer.step()

                    self._critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self._critic_optimizer.step()
                    
                    if self._wandb_flag:
                        wandb.log({'record/discrete_policy_loss': discrete_policy_loss.item(), 
                                'record/args_policy_loss': args_policy_loss.item(), 
                                'record/value_loss':value_loss.item(),
                                'record/discrete_entropy_loss': discrete_entropy_loss.item(),
                                'record/args_entropy_loss': args_entropy_loss.item(),
                                'record/dis_approx_kl:': dis_approx_kl,
                                'record/dis_clipfrac:': dis_clipfrac,
                                'record/args_approx_kl:': args_approx_kl,
                                'record/args_clipfrac:': args_clipfrac,
                                })
 

    def rollout(self, steps_per_epoch)-> None:
        '''
        Overview:
            Roll out to collect the sample and store to the buffer.
        Arguments:
            - env_id: the environment id. 
        '''
        # local_steps_per_epoch = 1000
        
        # env = gym.make(self._env_id)
        env = self._env

        # Prepare for interaction with environment
        start_time = time.time()
        obs, ep_ret, ep_len = env.reset(), 0, 0

        trajectory_obs = []
        trajectory_act_rew = []
        trajectory_obs_act_rew = []

        for t in range(steps_per_epoch):
            
                trajectory_obs = list(tuple(obs))
                # Get the discrete and parameters action, 
                with torch.no_grad():
                    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                    # get the logit of the action throught HPPO network
                    action_value = self._model.compute_actor_critic(state)

                    logit = action_value['logit']
                    value = action_value['value']
                    # discrete action
                    action_type_logit = logit['action_type']
                    prob = torch.softmax(action_type_logit, dim=-1)   # This 
                    pi_action = Categorical(prob)
                    action_type = pi_action.sample()                    
                    log_prob_action_type = pi_action.log_prob(action_type)

                    # continuous action
                    mu, sigma = logit['action_args']['mu'], logit['action_args']['sigma']
                    dist = Independent(Normal(mu, sigma), 1)
                    action_args = dist.sample()
                    # print('action_args:', action_args)
                    log_prob_action_args = dist.log_prob(action_args)

                    action = (int(action_type.cpu().numpy()), action_args.cpu().float().numpy().flatten())


                # interaction with the environment
                next_obs, reward, done, info = env.step(action)

                # record the action and reward for the trajectory.
                trajectory_act = [action[0], action[1][0]]
                trajectory_rew = [
                    info['reward_target'],      
                    info['reward_crash'],
                    info['reward_distance'],
                    info['reward_direction'],
                    info['reward']
                ]
                trajectory_act_rew.append(tuple(trajectory_act + trajectory_rew))
                trajectory_obs_act_rew.append(tuple(trajectory_obs + trajectory_act + trajectory_rew))

                ep_ret += reward
                ep_len += 1

                # Store the sample to the buffer.
                self._buf.store(
                    obs = obs,
                    next_obs = next_obs,
                    discrete_act = action_type.cpu(),
                    parameter_act = action_args.cpu(),
                    rew = reward,
                    val = value,
                    logp_discrete_act = log_prob_action_type.cpu(),
                    logp_parameter_act = log_prob_action_args.cpu(),
                    done = done,
                    logit_action_type = logit['action_type'].cpu(),
                    logit_action_argsmu = logit['action_args']['mu'].cpu(),
                    logit_action_argssigma = logit['action_args']['sigma'].cpu()
                )

                if self._wandb_flag:
                    wandb.log({'reward/rew': reward, 
                                'reward/value':value,
                              })

                # Update the obs
                obs = next_obs

                # The stop condition for each epoch
                epoch_ended = t==steps_per_epoch-1

                # The trajectory or epoch is stop
                if done or epoch_ended:
                    if reward!=0:
                        value = reward
                    else:
                        with torch.no_grad():
                            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                            # get the logit of the action throught HPPO network
                            action_value = self._model.compute_actor_critic(state)  
                        value = action_value['value'].cpu().float().numpy()                    

                    self._buf.finish_path(value)

                    if self._wandb_flag:
                        wandb.log({'ep_ret': ep_ret, 
                                   'ep_len':ep_len})
                        
                    ''' 
                    Record the trajectory data.
                    '''
                    # set the directory and file name
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_time_dir = f"{current_time}"

                    # Define the directory and file name
                    directory = 'log_3'
                    obs_act_rew_filename = 'obs_act_rew_'+current_time_dir+'.csv'

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Full file path for the observation，action and reward trajectory
                    obs_act_rew_filepath = os.path.join(directory, obs_act_rew_filename)

                    # if ep_ret > 0 and len(trajectory_obs_act_rew)>1:
                    if info['reward_target'] > 0 and len(trajectory_obs_act_rew)>1:
                        trajectory_final_obs = list(tuple(next_obs))
                        trajectory_obs_act_rew.append(tuple(trajectory_final_obs + trajectory_act + trajectory_rew))
                        # Write data to a CSV file
                        with open(obs_act_rew_filepath, 'w', newline='') as file:
                            writer = csv.writer(file)
                            # Write the header
                            writer.writerow(['pos_x', 'pos_y', 'pos_dir', 
                                            'target_x', 'target_y', 'target_dir',
                                            'bias_x', 'bias_y', 'bias_dir',
                                            'vert_dis1', 'vert_dis2', 'vert_dis3', 'vert_dis4',
                                            'action_type', 'action_args',
                                            'reward_target', 'reward_crash', 'reward_distance',
                                            'reward_direction', 'reward'])
                            writer.writerows(trajectory_obs_act_rew)


                    obs, ep_ret, ep_len = env.reset(), 0, 0


    def evaluate(self, eval_epoch=5):
        '''
        Eval the model 
        '''
        print('=====evaluate======')
        success_ratio = 0
        env = self._env
        mean_ep_ret, mean_ep_len = 0, 0

        obs, ep_ret, ep_len = env.reset(eval_stage=True), 0, 0

        eval_trajectory_act_rew = []
        eval_trajectory_obs_act_rew = []

        eval_epoch_step = eval_epoch

        # for t in range(eval_epoch):
        while eval_epoch_step > 0:                
            trajectory_obs = list(tuple(obs))
            # Get the discrete and parameters action, 
            with torch.no_grad():
                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                # get the logit of the action throught HPPO network
                action_value = self._model.compute_actor(state)

                logit = action_value['logit']
                # discrete action
                action_type_logit = logit['action_type']
                prob = torch.softmax(action_type_logit, dim=-1)        # This
                action_type = torch.argmax(prob, dim=1, keepdim=True) 
                # print('action_type:', action_type)

                # continuous action
                mu, _ = logit['action_args']['mu'], logit['action_args']['sigma']
                # print('mu:', mu)
                action_args = mu


                action = (int(action_type.cpu().numpy()), action_args.cpu().float().numpy().flatten())

            # interaction with the environment
            next_obs, reward, done, info = env.step(action)
            

            # record the action and reward for the trajectory.
            trajectory_act = [action[0], action[1][0]]
            trajectory_rew = [
                info['reward_target'],      
                info['reward_crash'],
                info['reward_distance'],
                info['reward_direction'],
                info['reward']
            ]
            eval_trajectory_act_rew.append(tuple(trajectory_act + trajectory_rew))
            eval_trajectory_obs_act_rew.append(tuple(trajectory_obs + trajectory_act + trajectory_rew))

            ep_ret += reward
            ep_len += 1

            # Update the obs
            obs = next_obs

            # The trajectory or epoch is stop
            if done :
                ''' 
                Record the trajectory data.
                '''
                # set the directory and file name
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_time_dir = f"{current_time}"

                # Define the directory and file name
                directory = 'eval_log_3'
                obs_act_rew_filename = 'obs_act_rew_'+current_time_dir+'.csv'

                # Create the directory if it doesn't exist
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Full file path for the observation，action and reward trajectory
                obs_act_rew_filepath = os.path.join(directory, obs_act_rew_filename)

                # if ep_ret > 0 and len(eval_trajectory_obs_act_rew)>1:
                if info['reward_target'] > 0 and len(eval_trajectory_obs_act_rew)>1:
                    trajectory_final_obs = list(tuple(next_obs))
                    eval_trajectory_obs_act_rew.append(tuple(trajectory_final_obs + trajectory_act + trajectory_rew))
                    # Write data to a CSV file
                    with open(obs_act_rew_filepath, 'w', newline='') as file:
                        writer = csv.writer(file)
                        # Write the header
                        writer.writerow(['pos_x', 'pos_y', 'pos_dir', 
                                        'target_x', 'target_y', 'target_dir',
                                        'bias_x', 'bias_y', 'bias_dir',
                                        'vert_dis1', 'vert_dis2', 'vert_dis3', 'vert_dis4',
                                        'action_type', 'action_args',
                                        'reward_target', 'reward_crash', 'reward_distance',
                                        'reward_direction', 'reward'])
                        writer.writerows(eval_trajectory_obs_act_rew)

                if info['reward_target'] > 0:
                    print('ep_ret:', ep_ret)
                    print('ep_len:', ep_len)
                

                mean_ep_ret += ep_ret
                mean_ep_len += ep_len

                if info['reward_target']> 0:
                    success_ratio += 1

                if self._wandb_flag:
                    wandb.log({'eval_ep_ret': ep_ret, 'eval_ep_len': ep_len})                

                obs, ep_ret, ep_len = env.reset(eval_stage = True), 0, 0
                eval_trajectory_act_rew = []
                eval_trajectory_obs_act_rew = []
                eval_epoch_step -= 1

        if self._wandb_flag:
            wandb.log({'eval_mean_ep_ret': mean_ep_ret/eval_epoch, 'eval_mean_ep_len':mean_ep_len/eval_epoch})
            wandb.log({'eval_success_ratio': success_ratio/eval_epoch})

        return mean_ep_ret/eval_epoch

    


        
        




    




                
                


























