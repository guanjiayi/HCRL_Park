import torch
import torch.nn as nn 
from easydict import EasyDict
import numpy as np

from typing import Union, Dict, Optional, TypeVar, List, Tuple, Callable
from collections import namedtuple


SequenceType = TypeVar('SequenceType', List, Tuple, namedtuple)



def fc_block(
        in_channels: int,
        out_channels: int,
        activation: nn.Module = None,
        norm_type: str = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
) -> nn.Sequential:
    """
    Overview:
        Create a fully-connected block with activation, normalization, and dropout.
        Optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): Type of the normalization.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block. Default is False.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the fully-connected block.

    """
    block = []
    block.append(nn.Linear(in_channels, out_channels))
    if norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:

    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN', 'SyncBN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN1': nn.InstanceNorm1d,
        'IN2': nn.InstanceNorm2d,
        'SyncBN': nn.SyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
    

def sequential_pack(layers: List[nn.Module]) -> nn.Sequential:

    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False
):
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)


class RunningMeanStd(object):
    """
    Overview:
       Wrapper to update new variable, new mean, and new count
    Interfaces:
        ``__init__``, ``update``, ``reset``, ``new_shape``
    Properties:
        - ``mean``, ``std``, ``_epsilon``, ``_shape``, ``_mean``, ``_var``, ``_count``
    """

    def __init__(self, epsilon=1e-4, shape=(), device=torch.device('cpu')):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate  \
                signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - epsilon (:obj:`Float`): the epsilon used for self for the std output
            - shape (:obj: `np.array`): the np array shape used for the expression  \
                of this wrapper on attibutes of mean and variance
        """
        self._epsilon = epsilon
        self._shape = shape
        self._device = device
        self.reset()

    def update(self, x):
        """
        Overview:
            Update mean, variable, and count
        Arguments:
            - ``x``: the batch
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties: ``_mean``, ``_var``, ``_count``
        """
        if len(self._shape) > 0:
            self._mean = np.zeros(self._shape, 'float32')
            self._var = np.ones(self._shape, 'float32')
        else:
            self._mean, self._var = 0., 1.
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        """
        Overview:
            Property ``mean`` gotten  from ``self._mean``
        """
        if np.isscalar(self._mean):
            return self._mean
        else:
            return torch.FloatTensor(self._mean).to(self._device)

    @property
    def std(self) -> np.ndarray:
        """
        Overview:
            Property ``std`` calculated  from ``self._var`` and the epsilon value of ``self._epsilon``
        """
        std = np.sqrt(self._var + 1e-8)
        if np.isscalar(std):
            return std
        else:
            return torch.FloatTensor(std).to(self._device)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


class DiscreteHead(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        dropout: Optional[float] = None,
        noise: Optional[bool] = False,
    ) -> None:

        super(DiscreteHead, self).__init__()
        layer = nn.Linear
        block = fc_block
        self.Q = nn.Sequential(
            MLP(
                hidden_size,
                hidden_size,
                hidden_size,
                layer_num,
                layer_fn=layer,
                activation=activation,
                use_dropout=dropout is not None,
                dropout_probability=dropout,
                norm_type=norm_type
            ), block(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        
        logit = self.Q(x)
        return {'logit': logit}
    

class ReparameterizationHead(nn.Module):

    default_sigma_type = ['fixed', 'independent', 'conditioned', 'happo', 'constent']
    default_bound_type = ['tanh', 'sigmoid', None]

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_num: int = 2,
            sigma_type: Optional[str] = None,
            fixed_sigma_value: Optional[float] = 1.0,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            bound_type: Optional[str] = None,
            hidden_size: int = None
    ) -> None:

        super(ReparameterizationHead, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.sigma_type = sigma_type
        assert sigma_type in self.default_sigma_type, "Please indicate sigma_type as one of {}".format(
            self.default_sigma_type
        )
        self.bound_type = bound_type
        assert bound_type in self.default_bound_type, "Please indicate bound_type as one of {}".format(
            self.default_bound_type
        )
        self.main = MLP(input_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.mu = nn.Linear(hidden_size, output_size)
        if self.sigma_type == 'fixed':
            self.sigma = torch.full((1, output_size), fixed_sigma_value)
        elif self.sigma_type == 'independent':  # independent parameter
            self.log_sigma_param = nn.Parameter(torch.zeros(1, output_size))
        elif self.sigma_type == 'conditioned':
            self.log_sigma_layer = nn.Linear(hidden_size, output_size)
        elif self.sigma_type == 'happo':
            self.sigma_x_coef = 1.
            self.sigma_y_coef = 0.5
            # This parameter (x_coef, y_coef) refers to the HAPPO paper http://arxiv.org/abs/2109.11251.
            self.log_sigma_param = nn.Parameter(torch.ones(1, output_size) * self.sigma_x_coef)

    def forward(self, x: torch.Tensor) -> Dict:

        x = self.main(x)
        mu = self.mu(x)
        if self.bound_type == 'tanh':
            mu = torch.tanh(mu)
        if self.sigma_type == 'fixed':
            sigma = self.sigma.to(mu.device) + torch.zeros_like(mu)  
        elif self.sigma_type == 'independent':
            log_sigma = self.log_sigma_param + torch.zeros_like(mu)  
            sigma = torch.exp(log_sigma)
        elif self.sigma_type == 'conditioned':
            log_sigma = self.log_sigma_layer(x)
            sigma = torch.exp(torch.clamp(log_sigma, -20, 2))
        elif self.sigma_type == 'happo':
            log_sigma = self.log_sigma_param + torch.zeros_like(mu)
            sigma = torch.sigmoid(log_sigma / self.sigma_x_coef) * self.sigma_y_coef
        return {'mu': mu, 'sigma': sigma}


class FCEncoder(nn.Module):
    """
    Overview:
        The full connected encoder is used to encode 1-dim input variable.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            activation: Optional[nn.Module] = nn.ReLU(),
            dropout: Optional[float] = None
    ) -> None:
        """
        Overview:
            Initialize the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent FC layers.
            - res_block (:obj:`bool`): Whether use ``res_block``. Default is ``False``.
            - activation (:obj:`nn.Module`): Type of activation to use in ``ResFCBlock``. Default is ``nn.ReLU()``.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResFCBlock`` \
                for more details. Default is ``None``.
            - dropout (:obj:`float`): Dropout rate of the dropout layer. If ``None`` then default no dropout layer.
        """
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])
 
        layers = []
        for i in range(len(hidden_size_list) - 1):
            layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            layers.append(self.act)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output embedding tensor of the env observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - x : :math:`(B, M)`, where ``M = obs_shape``.
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]``.
        Examples:
            >>> fc = FCEncoder(
            >>>    obs_shape=4,
            >>>    hidden_size_list=[32, 64, 64, 128],
            >>>    activation=nn.ReLU(),
            >>>    norm_type=None,
            >>>    dropout=None
            >>> )
            >>> x = torch.randn(1, 4)
            >>> output = fc(x)
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x
    

class RegressionHead(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_num: int = 2,
        final_tanh: Optional[bool] = False,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        hidden_size: int = None
    ) -> None:

        super(RegressionHead, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.main = MLP(input_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.last = nn.Linear(hidden_size, output_size)  # for convenience of special initialization

        self.final_tanh = final_tanh
        if self.final_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict:

        x = self.main(x)
        x = self.last(x)
        if self.final_tanh:
            x = self.tanh(x)
        if x.shape[-1] == 1 and len(x.shape) > 1:
            x = x.squeeze(-1)
        return {'pred': x}


class HCRL_Model(nn.Module):
    '''
    Define the HPPO network
    '''
    def __init__(
            self,
            obs_shape: int,
            discrete_act_dim: int,
            parameter_act_dim: int,
            share_encoder: bool = True,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            sigma_type: Optional[str] = 'independent',
            fixed_sigma_value: Optional[int] = 0.3,
            bound_type: Optional[str] = None,
    )-> None:
        super(HCRL_Model, self).__init__()
        self.share_encoder = share_encoder

        '''
        The critic network
        '''
        # Encoder network
        if self.share_encoder:
            self.encoder = FCEncoder(
            obs_shape = obs_shape,
            hidden_size_list = encoder_hidden_size_list,
            activation = activation,
            )
        else :
            self.actor_encoder = FCEncoder(
            obs_shape = obs_shape,
            hidden_size_list = encoder_hidden_size_list,
            activation = activation,
            )
            self.critic_encoder = FCEncoder(
            obs_shape = obs_shape,
            hidden_size_list = encoder_hidden_size_list,
            activation = activation,
            )


        # The head of the critic network
        self.critic_head = RegressionHead(
            input_size = encoder_hidden_size_list[-1],
            output_size = 1,
            layer_num = critic_head_layer_num,
            activation = activation,
            norm_type = norm_type,
            hidden_size = critic_head_hidden_size
        )

        '''
        The actor network
        '''
        # The head of the parameters action network
        actor_action_args = ReparameterizationHead(
            input_size=encoder_hidden_size_list[-1],
            output_size=parameter_act_dim,
            layer_num=actor_head_layer_num,
            sigma_type=sigma_type,
            fixed_sigma_value=fixed_sigma_value,
            activation=activation,
            norm_type=norm_type,
            bound_type=bound_type,
            hidden_size=actor_head_hidden_size,
        )
        # The head of the discrete action network
        actor_action_type = DiscreteHead(
            hidden_size = actor_head_hidden_size,
            output_size = discrete_act_dim,
            layer_num=actor_head_layer_num,
            activation=activation,
            norm_type=norm_type,
        )
        # The head of the parameters and discrete action network
        self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        '''
        The actor and critic network
        '''
        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)



    def compute_actor(self, x: torch.Tensor) -> Dict:

        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.actor_encoder(x)
        # x = self.encoder(x)

        action_type = self.actor_head[0](x)
        action_args = self.actor_head[1](x)
        return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}}
        

    def compute_critic(self, x: torch.Tensor) -> Dict:

        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.critic_encoder(x)

        x = self.critic_head(x)
        return {'value': x['pred']}
    
    def compute_actor_critic(self, x: torch.Tensor) -> Dict:

        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(x)
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)

        value = self.critic_head(critic_embedding)['pred']

        action_type = self.actor_head[0](actor_embedding)
        action_args = self.actor_head[1](actor_embedding)
        
        output = {
            'logit': {'action_type': action_type['logit'], 'action_args': action_args}, 
            'value': value
            }
        return output        

    


        