# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import spectral_norm, weight_norm

# Utils
torch.manual_seed(0)

def clever_single_action(env, action_batch, obs_indices, type1=None, type2=None):
    if action_batch is None:
        return None
    
    # By default, random actions
    # Pop 0 is random
    # Pop 1 is semi-random
    # Pop 2 is random
    actions = env.configurable_actions_grid(None, type1, type2)
    
    batch_idx, x_idx, y_idx = obs_indices.T

    # Now assign:
    # 0: move to clever
    actions[batch_idx, x_idx, y_idx] = action_batch.float()
    
    return actions


# Simple policy network for a discrete action space (0 to 16):
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(obs_dim, hidden_dim))
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        # (optional) input check omitted for brevity
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        x = F.relu(z)

        x = self.dropout(x)

        logits = self.fc2(x)

        # final safety: make it a valid distribution
        probs = F.softmax(logits, dim=-1).clamp(min=1e-8, max=1-1e-8)
        return probs

    def __init__(self, obs_shape, num_actions=17):        
        super().__init__()
        
        self.saved_log_probs = []
        self.rewards = []
        self.exploration_importance = 0.01
        self.observation_indices = []

        self.saved_entropies = []
        
        self.all_rewards = []
        self.all_losses = []
        self.action_counts = [0] * num_actions
        
        C, H, W = obs_shape
        # Flatten the observation
        self.flatten = nn.Flatten()
        self.fc1 = weight_norm(nn.Linear(C * H * W, 100))

        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = weight_norm(nn.Linear(100, num_actions))
        
    def forward(self, x):
        # x shape: (batch, C, H, W)
        if torch.isnan(x).any():
            print("NaN in x before flatten")
            print(x)
            raise ValueError("NaN in x before flatten")

        x = x.view(x.size(0), -1)  # Flatten the input

        if torch.isnan(x).any():
            print("NaN in x before fc1")
            print(x)
            raise ValueError("NaN in x before fc1")

        x = torch.relu(self.fc1(x))

        if torch.isnan(x).any():
            print("NaN in x after fc1")
            print(x)
            raise ValueError("NaN in x after fc1")

        x = self.dropout(x)  # Apply dropout

        if torch.isnan(x).any():
            print("NaN in x after fc1 + dropout")
            print(x)
            raise ValueError("NaN in x after fc1 + dropout")

        logits = self.fc2(x)

        # Check nan
        if torch.isnan(logits).any():
            print("NaN in logits")
            print(x)
            print(logits)
            raise ValueError("NaN in logits")

        probs = F.softmax(logits, dim=-1).clamp(min=1e-8, max=1-1e-8)  # Convert logits to probabilities

        return probs


def select_action(policy_net, observations):
    if observations.size(0) == 0:
        return None

    probs = policy_net(observations)     # shape: (batch, n_actions)
    m     = Categorical(probs)

    action = m.sample()                  # shape: (batch,)
    policy_net.saved_log_probs.append(m.log_prob(action))  # for PG loss
    policy_net.saved_entropies.append(m.entropy())

    # optional: track counts for diagnostics
    unique, counts = torch.unique(action, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        policy_net.action_counts[u] += c / 10000

    return action

def finish_episode(policy_net, optimizer, gamma=0.99, max_grad_norm=1.0):
    if not policy_net.saved_log_probs:
        return

    device = policy_net.saved_log_probs[0].device
    T      = len(policy_net.rewards)

    # ---------- 2.1  gather all agent‑ids once, build mapping ----------
    # cat →  (∑_t n_t, 3)
    flat_ids      = torch.cat(policy_net.observation_indices, dim=0)
    # unique over rows gives:  unique_ids … (N,3),  inverse_map … (∑_t n_t,)
    unique_ids, inverse_map = torch.unique(flat_ids,
                                           return_inverse=True,
                                           dim=0)
    N_agents = unique_ids.size(0)

    # how many agents appear at each timestep?
    lens   = torch.tensor([ids.size(0) for ids in policy_net.observation_indices],
                          device=device)                                # (T,)
    # cumulative lengths let us slice inverse_map without Python loops
    # ptr[t] = start index for timestep t inside inverse_map
    ptr = torch.cat((lens.new_zeros(1), lens.cumsum(0)), dim=0)         # (T+1,)

    running_return = torch.zeros(N_agents, device=device)
    returns        = [None] * T            # will fill from back to front

    # ---------- 2.2  backwards pass over timesteps (no inner loop) ----
    for t in range(T - 1, -1, -1):
        idx_slice   = inverse_map[ptr[t]:ptr[t+1]]      # (n_t,)  long / int64
        step_return = policy_net.rewards[t].to(torch.float32)

        # vectorised Bellman update:
        #   R_a ← r_t(a) + γ·R_a     ∀ agents present at this timestep
        running_return.index_put_((idx_slice,),
                                  step_return + gamma * running_return[idx_slice],
                                  accumulate=False)

        returns[t] = running_return[idx_slice]           # (n_t,)

    # ---------- 2.3  flatten, standardise, loss -----------------------
    flat_returns   = torch.cat(returns,        dim=0)             # matches order
    flat_log_probs = torch.cat(policy_net.saved_log_probs, dim=0)

    # --- flatten the saved entropies ---
    flat_entropies = torch.cat(policy_net.saved_entropies, dim=0)

    # --- standardize returns as before ---
    flat_returns = (flat_returns - flat_returns.mean()) / (flat_returns.std() + 1e-6)

    # --- combine PG loss with entropy bonus ---
    pg_loss      = -(flat_log_probs * flat_returns).sum()
    entropy_loss = -policy_net.exploration_importance * flat_entropies.sum()    # note the minus: we want to *maximize* entropy
    policy_loss  = pg_loss + entropy_loss

    policy_net.exploration_importance *= 0.99


    # ---------- 2.4  optimise ----------------------------------------
    optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimizer.step()

    # ---------- 2.5  bookkeeping -------------------------------------
    policy_net.all_losses.append(policy_loss.item())
    episode_return = sum(r.sum().item() for r in policy_net.rewards)
    policy_net.all_rewards.append(episode_return)

    policy_net.rewards.clear()
    policy_net.saved_log_probs.clear()
    policy_net.observation_indices.clear()
    policy_net.saved_entropies.clear()


