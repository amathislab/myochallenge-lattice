import torch
from torch import nn
from typing import Dict, Type, Union, Tuple
from stable_baselines3.common.utils import get_device


class TransformerExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict[str, int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.shared_net = nn.Identity().to(device)
        self.policy_net = self.create_transformer_network(feature_dim, activation_fn, net_arch.get("pi")).to(device)
        self.value_net = self.create_transformer_network(feature_dim, activation_fn, net_arch.get("vf")).to(device)
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim
        self._reset_parameters()

    def create_transformer_network(self, feature_dim, activation_fn, net_features):
        if net_features is None:
            return nn.Identity()
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                activation=activation_fn(),
                nhead=net_features["num_heads"],
                batch_first=True,
                dim_feedforward=net_features["dim_feedforward"],
                dropout=net_features["dropout"],
                layer_norm_eps=net_features["layer_norm_eps"],
                norm_first=net_features["norm_first"],
            )
            layer_norm = nn.LayerNorm(
                feature_dim, eps=net_features["layer_norm_eps"]
            )
            return nn.TransformerEncoder(transformer_layer, net_features["num_layers"], layer_norm)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))
    
    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
