import warnings
import gym
import torch
from typing import Any, Dict, List, Optional, Type, Union
from torch import nn
from models.distributions import (
    LatticeNoiseDistribution,
    PerMuscleSquashedDiagGaussianDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import SACPolicy, Actor
from models.feature_extractors import TransformerFeaturesExtractor


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MuscleTransformerActor(Actor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        **unused_kwargs
    ):
        print("INFO: unused actor kwargs", unused_kwargs)
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )

        if sde_net_arch is not None:
            warnings.warn(
                "sde_net_arch is deprecated and will be removed in SB3 v2.4.0.",
                DeprecationWarning,
            )

        action_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            raise NotImplementedError()
        else:
            self.action_dist = PerMuscleSquashedDiagGaussianDistribution(action_dim)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                last_layer_dim, log_std_init
            )


class MuscleTransformerCritic(ContinuousCritic):
    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = torch.cat([features.mean(dim=1), actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features.mean(dim=1), actions], dim=1))


class MuscleTransformerSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -3,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        **unused_kwargs
    ):
        print("INFO: unused kwargs", unused_kwargs)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            log_std_init=log_std_init,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> MuscleTransformerActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return MuscleTransformerActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> MuscleTransformerCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return MuscleTransformerCritic(**critic_kwargs).to(self.device)


class LaticeActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_latice=False,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        self.use_latice = use_latice
        self.std_clip = std_clip
        self.expln_eps = expln_eps
        self.std_reg = std_reg
        self.alpha = alpha
        if use_latice:
            assert self.use_sde
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
            action_dim = get_action_dim(self.action_space)
            self.action_dist = LatticeNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                squash_output=True,
                learn_features=True,
                epsilon=expln_eps,
                std_clip=std_clip,
                std_reg=std_reg,
                alpha=alpha,
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean)
                )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                use_latice=self.use_latice,
                std_clip=self.std_clip,
                expln_eps=self.expln_eps,
                std_reg=self.std_reg,
                alpha=self.alpha,
            )
        )
        return data

    def get_std(self) -> torch.Tensor:
        std = super().get_std()
        if self.use_latice:
            std = torch.cat(std, dim=1)
        return std


class LaticeSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_latice=True,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        use_sde=False,
        alpha=1,
        **kwargs
    ):
        super().__init__(
            observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs
        )
        self.latice_kwargs = {
            "use_latice": use_latice,
            "expln_eps": expln_eps,
            "std_clip": std_clip,
            "std_reg": std_reg,
            "alpha": alpha,
        }
        self.actor_kwargs.update(self.latice_kwargs)
        if use_latice:
            assert use_sde
            self._build(lr_schedule)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return LaticeActor(**actor_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(self.latice_kwargs)
        return data
