import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from models.helpers import SinCosPositionalEncoding, LearnedPositionalEncoding
from definitions import ACT_KEY, OBJ_KEY, GOAL_KEY, POSITIONS_KEY


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embedding_size=16,
        num_heads=4,
        num_layers=4,
        layer_norm_eps=1e-5,
        dim_feedforward=32,
        dropout=0.1,
        position_embedding="learned",  # "sin_cos" or "learned"
        norm_first=False,
        num_tokens=100,
    ):
        super().__init__(observation_space=observation_space, features_dim=1)
        self._features_dim = embedding_size

        random_obs = observation_space.sample()

        # Define the linear embeddings for the actuator features
        act_obs = random_obs.get(ACT_KEY)
        if act_obs is None:
            raise ValueError("ERROR: there seems to be no actuator")
        else:
            num_timesteps, num_act_features, _ = random_obs[ACT_KEY].shape
            self.actuator_encoder = nn.Linear(
                in_features=num_timesteps * num_act_features,
                out_features=embedding_size,
            )

        obj_obs = random_obs.get(OBJ_KEY)
        if obj_obs is not None:
            num_obj_features = obj_obs.shape[1]
            self.object_encoder = nn.Linear(
                in_features=num_timesteps * num_obj_features,
                out_features=embedding_size,
            )

        goal_obs = random_obs.get(GOAL_KEY)
        if goal_obs is not None:
            num_goal_features = goal_obs.shape[1]
            self.goal_encoder = nn.Linear(
                in_features=num_timesteps * num_goal_features,
                out_features=embedding_size,
            )
        self.num_tokens = num_tokens

        if position_embedding == "sin_cos":
            self.positional_encoder = SinCosPositionalEncoding(
                num_tokens=self.num_tokens, d_model=embedding_size, dropout=dropout
            )
        elif position_embedding == "learned":
            self.positional_encoder = LearnedPositionalEncoding(
                num_tokens=self.num_tokens, d_model=embedding_size, dropout=dropout
            )
        else:
            raise ValueError("Unknown position embedding type:", position_embedding)
        
        if num_layers == 0:
            self.encoder = nn.Identity()
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                batch_first=True,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
            )
            layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
            self.encoder = nn.TransformerEncoder(transformer_layer, num_layers, layer_norm)

        self._reset_parameters()

    def forward(self, observation_dict):
        embedding_list = []
        actuator_features = observation_dict[ACT_KEY]  # (batch, time, feat, act)
        num_actuators = actuator_features.shape[-1]
        actuator_features = actuator_features.transpose(1, -1).flatten(
            start_dim=2
        )  # (batch, act, time*feat)
        actuator_embeddings = self.actuator_encoder(actuator_features)
        embedding_list.append(actuator_embeddings)

        object_features = observation_dict.get(OBJ_KEY)
        if object_features is not None:
            object_features = object_features.transpose(1, -1).flatten(start_dim=2)
            object_embedding = self.object_encoder(object_features)
            embedding_list.append(object_embedding)

        goal_features = observation_dict.get(GOAL_KEY)
        if goal_features is not None:
            goal_features = goal_features.transpose(1, -1).flatten(start_dim=2)
            goal_embedding = self.goal_encoder(goal_features)
            embedding_list.append(goal_embedding)

        ## TODO: complete with the encoder block
        positions = observation_dict.get(POSITIONS_KEY)
        input_embeddings = self.positional_encoder(
            torch.concat(embedding_list, axis=1), positions
        )
        encodings = self.encoder(input_embeddings)
        return encodings[:, :num_actuators, :]  # Only includes the actuator encodings

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
