import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os

# Constants from engine
class Map:
    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4
    BOMB = 5

class Player:
    MAX_BOMB_RADIUS = 5
    MAX_BOMB_CAPACITY = 5

BOMB_MAX_TIMER = 7

# DQN Model Architecture
class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    """
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)

def encode_obs(obs, agent_id):
    """
    Convert raw observation to (map_feat, aux_feat) for DQN model.
    
    Args:
        obs: dict with keys 'map', 'players', 'bombs'
        agent_id: int, this agent's player ID
    
    Returns:
        map_feat: (9, 13, 13) float32 tensor
        aux_feat: (3,) float32 array
    """
    if obs is None:
        raise ValueError("obs should not be None")

    grid = obs["map"]      # (H, W)
    players = obs["players"]  # (num_players, 5)
    bombs = obs["bombs"]    # (N, 4), N may be 0
    H, W = grid.shape

    # One-hot map: grass, wall, box, item_radius, item_capacity
    map_channels = []
    for v in [Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        map_channels.append((grid == v).astype(np.float32))

    # Player position masks
    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[agent_id]
    
    # Assume 1v1: opponent is the other player
    opp_id = 1 - agent_id
    ox, oy, opp_alive, _, _ = players[opp_id]
    
    my_pos = np.zeros((H, W), dtype=np.float32)
    opp_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive) == 1:
        my_pos[int(my_x), int(my_y)] = 1.0
    if int(opp_alive) == 1:
        opp_pos[int(ox), int(oy)] = 1.0

    # Bomb channels
    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        t = float(timer) / BOMB_MAX_TIMER  # normalise by default max timer
        bomb_timer[bx, by] = max(bomb_timer[bx, by], t)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == agent_id else 0.0

    scalar = np.array([
        float(my_bombs_left) / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        float(opp_alive),
    ], dtype=np.float32)

    map_feat = np.stack([
        *map_channels,          # 5 channels
        my_pos,                 # 1 channel
        opp_pos,                # 1 channel
        bomb_timer,             # 1 channel
        bomb_owned,             # 1 channel
    ], axis=0).astype(np.float32)  # (9, H, W)
    
    return map_feat, scalar

class Agent:
    """DQN Agent for submission."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")  # Use CPU for compatibility
        self.q_net = None
        self.map_shape = ((9, 13, 13))
        self.aux_dim = 3
        self.num_actions = 6
        
        # Load checkpoint from same directory as this file
        checkpoint_path = Path(__file__).parent / "2737502_global_step.pth"
        self._load_checkpoint(str(checkpoint_path))
    
    def _load_checkpoint(self, checkpoint_path):
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get input spec from checkpoint
            input_spec = checkpoint.get("input_spec", 
                                       checkpoint.get("input_shape", 
                                                     checkpoint["input_dim"]))
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.num_actions = checkpoint["num_actions"]
            
            # Create and load model
            self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions)
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.q_net.to(self.device)
            self.q_net.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            raise
    
    def act(self, obs):
        """
        Take an action based on observation.
        
        Args:
            obs: dict with keys 'map', 'players', 'bombs'
        
        Returns:
            action: int in range [0, 5]
        """
        try:
            # Encode observation
            map_state, aux_state = encode_obs(obs, self.agent_id)
            
            # Convert to tensors and add batch dimension
            map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
            aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
            
            # Get Q-values and select best action
            with torch.no_grad():
                q_values = self.q_net(map_tensor, aux_tensor)
                action = q_values.argmax(dim=1).item()
            
            return action
        except Exception as e:
            print(f"[ERROR] Agent.act() failed: {e}")
            # Fallback to random action on error
            return 0
