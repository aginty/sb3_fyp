from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod
import copy

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, BaseContinuousCritic, MlpContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim, get_obs_shape
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule


class BaseActor(BasePolicy, ABC):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: Optional[nn.Module] = None,
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        squash_output: bool = True
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            squash_output=True
        )

        self.features_dim = self.get_features_dim()
        self.action_dim = get_action_dim(self.action_space)
        
        # Deterministic action
        self.mu = self.build_mu()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                features_extractor_class=features_extractor_class,
                features_extractor_kwargs=features_extractor_kwargs
            )
        )
        return data
    
    @abstractmethod
    def get_features_dim(self):
        """method for getting the feature dimension"""

    @abstractmethod
    def build_mu(self) -> nn.Module:
        """method for creating neural network function approximator"""

    @abstractmethod
    def forward(self, features: th.Tensor) -> th.Tensor:
        """method to pass data through network"""

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        if self.has_feature_extractor():
            features = self.extract_features(observation)
        else:
            features = observation
        return self(features) #change back to predict if this causes error

class MlpActor(BaseActor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor: Optional[nn.Module] = None,
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        is_image: bool = False
    ):

        if net_arch is None:
            if is_image:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images= normalize_images,
            squash_output=True
        )


    def get_features_dim(self):
        return get_flattened_obs_dim(self.observation_space) #observation_space = spaces.Box(-inf, +inf, shape=(self.state_space,))
        

    def build_mu(self) -> nn.Module:
        print("features dim is", self.features_dim)
        actor_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn, squash_output=True)
        mu = nn.Sequential(*actor_net)
        return mu

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.mu(features)


from torch.nn import Module, Conv2d, Conv1d, Linear, MaxPool2d, ReLU, LogSoftmax, Dropout, Flatten, Tanh, Sequential
from torch import flatten

class CNNActor(BaseActor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor: Optional[nn.Module] = None,
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        is_image: bool = False
    ):

        self.activation_fn = activation_fn

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images= normalize_images,
            squash_output=True
        )

    #need to override this method defined in superclass BaseActor
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        # features = observation[0] #observation = state: st = [bt, cpt, ht, I], observation[0] = I
        return self(features)

    def get_features_dim(self):
        """
        features dim is the dimension of the technical indicators (num indicators + num days d)
        """
        return 15 #num_historic days -> don't think this will actually be used
        #return self.observation_space.shape #Box
    
    #CNN: 4 layers, 16 filters, 0 pool layers
    def build_mu(self) -> nn.Module:
        #CNN for 1 dimensional time series 
        actor_net = []
        actor_net.append(Conv1d(in_channels=1, out_channels=16, kernel_size=3)) #input size=num channels, output_size=num feature maps=num filters
        actor_net.append(ReLU())
        actor_net.append(Dropout(p=0.5))
        actor_net.append(Conv1d(in_channels=16, out_channels=16, kernel_size=3))
        actor_net.append(ReLU())
        actor_net.append(Dropout(p=0.5))
        actor_net.append(Flatten())

        actor_net.append(Linear(16, 400))
        actor_net.append(ReLU())
        actor_net.append(Linear(400, 300))
        actor_net.append(ReLU())
        actor_net.append(Linear(300, 1)) #single action - single stock trading
        actor_net.append(Tanh())

        mu = nn.Sequential(*actor_net)

        return mu


    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.mu(features)


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        Actor: Optional[Type[BaseActor]] = MlpActor,
        Critic: Optional[Type[BaseContinuousCritic]] = MlpContinuousCritic,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True
        )

        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space, action_space)

        # if actor is None:
        #     self.actor = MlpActor(observation_space, action_space)
        # else:
        #     self.actor = actor

        # if critic is None:
        #     self.critic = MlpContinuousCritic(observation_space, action_space)
        # else:
        #     self.critic = critic

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data


    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor._predict(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TD3Policy


class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
