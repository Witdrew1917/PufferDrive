from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv
from pufferlib.ocean.attention import AttentionBlock  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper

class BootstrappedDrive(nn.Module):

    def __init__(self, env, input_size=128, hidden_size=128, **kwargs) -> None:
        super().__init__()

        self.policy = Drive(env, input_size, hidden_size, **kwargs)
        self.anchor = BCPolicy(env, input_size, hidden_size, **kwargs)

    def forward(self, observations, state=None):
        return self.policy(observations)

    def forward_train(self, x, state=None):
        return self.policy.forward_train(x, state)

    def forward_eval(self, observations, state=None):
        return self.policy.forward_eval(observations)


class BCPolicy(nn.Module):
    """BC policy supporting both joint (single head) and independent (multi-head) action spaces."""

    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()

        output_sizes = env.single_action_space.nvec.tolist()
        self.num_heads = len(output_sizes)
        self.hidden_size = hidden_size
        self.obs_dim = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        self.ego_dim = env.ego_features
        self.is_continuous = False

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.shared_embedding = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, 512)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(512, hidden_size)),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, s) for s in output_sizes])

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        # Bounded so trailing features (waypoints, PID targets) don't leak into the road block
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        return F.relu(self.shared_embedding(concat_features))

    def dist(self, obs):
        hidden = self.encode_observations(obs.float())
        return [Categorical(logits=head(hidden)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        dists = self.dist(obs)
        actions = [d.logits.argmax(dim=-1) if deterministic else d.sample() for d in dists]
        actions = actions[0].float() if self.num_heads == 1 else [a.float() for a in actions]
        return actions

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations.float())
        actions = [head(hidden) for head in self.heads]
        return actions, torch.zeros(size=hidden.shape[:-1] + (1,), device=hidden.device)

    def get_action_dist_logits(self, obs):
        """Get logits from all heads. Returns list of tensors."""
        return [d.logits for d in self.dist(obs)]

    def _log_prob(self, obs, expert_actions):
        dists = self.dist(obs)
        if self.num_heads == 1:
            return dists[0].log_prob(expert_actions.squeeze(-1).long()).mean()
        log_prob = sum(d.log_prob(expert_actions[:, i].long()).mean() for i, d in enumerate(dists))
        return log_prob / self.num_heads

    def _entropy(self, obs):
        dists = self.dist(obs)
        if self.num_heads == 1:
            return dists[0].entropy().mean().item()
        return {f"entropy_head_{i}": d.entropy().mean().item() for i, d in enumerate(dists)}

class NaviPBoostedBCPolicy(BCPolicy):
    # Tells pufferl not to zero the PID target observations before the forward pass
    uses_pid_targets = True
    supports_recurrent = False

    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__(env, input_size, hidden_size, **kwargs)
        self.pid_target_slice = env.pid_target_slice
        if self.pid_target_slice is None or self.num_heads != 3:
            raise ValueError("NaviPBoostedBCPolicy requires PID targets and three discrete delta_local action heads")
        self.pid_anchor_smearing = env.pid_anchor_smearing

    def dist(self, obs):
        hidden = self.encode_observations(obs.float())
        actions = [head(hidden) for head in self.heads]
        actions = self.apply_navip(obs, actions)

        return [Categorical(logits=logits) for logits in actions]

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations.float())
        actions = [head(hidden) for head in self.heads]
        actions = self.apply_navip(observations, actions)
        return actions, torch.zeros(size=hidden.shape[:-1] + (1,), device=hidden.device)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def apply_navip(self, observations, actions) -> list[Tensor]:

        targets = observations[:, self.pid_target_slice].long()
        valid_targets = (targets >= 0).all(dim=-1)
        safe_targets = targets.masked_fill(~valid_targets[:, None], 0)  # -1 is not a valid index for one_hot
        navi_p_action = []
        for h, head in enumerate(actions):
            q = F.one_hot(safe_targets[:, h], num_classes=head.shape[-1]).float()      # [B, n_bins_h]
            q = pufferlib.pytorch.smear_categorical_targets(q, self.pid_anchor_smearing)
            q = q * valid_targets[:, None]  # no PID target -> fall back to the BC distribution
            p_bc = F.softmax(head.float(), dim=-1)
            w = 0.5 * valid_targets[:, None].float()
            navi_p_action.append(torch.log((1 - w) * p_bc + w * q + 1e-8))  # mix in probability space, return logits

        return navi_p_action


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        self.ego_dim = env.ego_features
        self.max_waypoints = env.max_waypoints
        self.waypoint_features = env.waypoint_features
        self.use_waypoints = self.max_waypoints > 0
        self.waypoint_encoder_arch = env.waypoint_encoder_arch

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        num_branches = 3
        if self.use_waypoints:
            if self.waypoint_encoder_arch == "mlp":
                self.waypoint_encoder = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(self.waypoint_features, input_size)),
                    nn.LayerNorm(input_size),
                    pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
                )

            elif self.waypoint_encoder_arch == "attention":
                self.waypoint_encoder = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(self.waypoint_features, input_size)),
                    AttentionBlock(input_size),
                )

            num_branches += 1

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(num_branches * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        branch_features = [ego_features, road_features, partner_features]
        if self.use_waypoints:
            waypoint_start = ego_dim + partner_dim + road_dim
            waypoint_obs = observations[
                :, waypoint_start : waypoint_start + self.max_waypoints * self.waypoint_features
            ]
            waypoints = waypoint_obs.view(-1, self.max_waypoints, self.waypoint_features)
            waypoint_features = self.waypoint_encoder(waypoints)
            if self.waypoint_encoder_arch == "mlp":
                waypoint_features, _ = waypoint_features.max(dim=1)
            elif self.waypoint_encoder_arch == "attention":
                waypoint_features = waypoint_features[:, 0]

            branch_features.append(waypoint_features)

        concat_features = torch.cat(branch_features, dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value
