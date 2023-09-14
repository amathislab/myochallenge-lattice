from typing import Any, Dict, List, Type, Optional, Union
import gym
import torch
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    create_mlp,
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.noise import ActionNoise


class VAEActor(BasePolicy):
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
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        embedding_dim=5,
        embedding_noise: Optional[ActionNoise] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.embedding_dim = embedding_dim
        self.embedding_noise = embedding_noise

        action_dim = get_action_dim(self.action_space)
        encoder_net = create_mlp(
            features_dim, embedding_dim, net_arch, activation_fn, squash_output=True
        )
        decoder_net = create_mlp(
            embedding_dim, action_dim, net_arch[::-1], activation_fn, squash_output=True
        )
        # Deterministic action
        self.encoder = nn.Sequential(*encoder_net)
        self.decoder = nn.Sequential(*decoder_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                embedding_dim=self.embedding_dim,
                embedding_noise=self.embedding_noise,
            )
        )
        return data

    def forward(self, obs: torch.Tensor, deterministic: bool=True) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        embedding = self.encoder(features)
        if not deterministic:
            embedding += torch.Tensor(self.embedding_noise()).to(embedding.device)
        return self.decoder(embedding)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation, deterministic)


class VariationalTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        embedding_dim: Optional[int] = 5,
        embedding_noise: Optional[ActionNoise] = None,
    ):
        self.embedding_noise = embedding_noise
        self.embedding_dim = embedding_dim
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Differently from standard TD3, here we do not ignore the deterministic parameter
        return self.actor(observation, deterministic)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> VAEActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs.update(
            {
                "embedding_noise": self.embedding_noise,
                "embedding_dim": self.embedding_dim,
            }
        )

        return VAEActor(**actor_kwargs).to(self.device)
