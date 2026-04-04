import os
import json
import importlib.util
from datetime import datetime

from visualizer.bomberland_rendering import render_match_frame
from engine.game import BomberEnv


class MatchRunner:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'gifs'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'json'), exist_ok=True)

    def load_agent(self, agent_path, agent_id):
        """Loads an agent from a file path.

        Supports both submission format (class Agent) and baseline agents
        (class names ending with Agent, for example SmarterRuleAgent).
        """
        try:
            spec = importlib.util.spec_from_file_location("Agent", agent_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec: {agent_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "Agent") and isinstance(getattr(module, "Agent"), type):
                agent_cls = getattr(module, "Agent")
            else:
                agent_cls = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and attr_name.endswith("Agent"):
                        agent_cls = attr
                        break

                if agent_cls is None:
                    raise AttributeError(
                        "No valid agent class found. Expected 'Agent' or a class ending with 'Agent'."
                    )

            try:
                return agent_cls(agent_id)
            except TypeError:
                return agent_cls()
        except Exception as e:
            print(f"Error loading agent from {agent_path} (agent_id={agent_id}): {e}")
            return None

    def run_match(self, agent_paths, team_ids, seed=None, max_steps=500):
        env = BomberEnv(seed=seed, max_steps=max_steps)
        agents = []
        for i, path in enumerate(agent_paths):
            agent = self.load_agent(path, i)
            agents.append(agent)
        
        obs = env.reset(seed=seed)
        frames = []
        history = []
        
        terminated = False
        truncated = False
        
        # Initial frame and history snapshot
        initial_obs = {
            "map": obs["map"].tolist(),
            "players": obs["players"].tolist(),
            "bombs": obs["bombs"].tolist(),
            "_step": env.current_step,
        }
        history.append({
            "step": env.current_step,
            "actions": None,
            "alive": [bool(p.alive) for p in env.players],
            "map": initial_obs["map"],
            "players": initial_obs["players"],
            "bombs": initial_obs["bombs"],
        })
        frames.append(render_match_frame(initial_obs, prev_obs=None))
        
        # Track death order for ranking
        death_order = []
        ranks = [0] * len(agents)
        alive_mask = [True] * len(agents)

        while not (terminated or truncated):
            prev_obs = obs
            actions = []
            for i, agent in enumerate(agents):
                if env.players[i].alive:
                    try:
                        # Pass observation to agent
                        action = agent.act(obs)
                    except Exception as e:
                        print(f"Agent {i} error: {e}")
                        action = 0 # Default to idle
                    actions.append(action)
                else:
                    actions.append(0)
            
            obs, terminated, truncated = env.step(actions)
            
            # Record state for JSON
            history.append({
                "step": env.current_step,
                "actions": actions,
                "alive": [bool(p.alive) for p in env.players],
                "map": obs["map"].tolist(),
                "players": obs["players"].tolist(),
                "bombs": obs["bombs"].tolist(),
            })
            
            # Render frame for GIF
            frame_obs = {
                "map": obs["map"].tolist(),
                "players": obs["players"].tolist(),
                "bombs": obs["bombs"].tolist(),
                "_step": env.current_step,
            }
            prev_frame_obs = {
                "map": prev_obs["map"].tolist(),
                "players": prev_obs["players"].tolist(),
                "bombs": prev_obs["bombs"].tolist(),
                "_step": env.current_step - 1,
            }
            frames.append(render_match_frame(frame_obs, prev_obs=prev_frame_obs))
            
            deaths = []
            for i, p in enumerate(env.players):
                if alive_mask[i] and not p.alive:
                    alive_mask[i] = False
                    deaths.append(i)
            # death_order = [[1, 2], [3]] meaning 1 and 2 died at the same time, then 3, 0 is still alive. Or [[1]] then only 1 died, 0, 2, 3 are still alive and they can all survive until the end and gets rank 0.
            if deaths:
                death_order.append(deaths)

        # determine who's alive = rank 0
        alives = []
        for i, p in enumerate(env.players):
            if alive_mask[i] and p.alive:
                alives.append(i)
                alive_mask[i] = False
        if alives:
            death_order.append(alives)
        
        # Determine final ranks, backward
        for rank, group in enumerate(reversed(death_order)):
            for i in group:
                ranks[i] = rank

        # Save logs
        match_name = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"
        
        # Save GIF
        gif_path = os.path.join(self.log_dir, 'gifs', f"{match_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
        
        # Save JSON
        json_path = os.path.join(self.log_dir, 'json', f"{match_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "seed": seed,
                "team_ids": team_ids,
                "ranks": ranks,
                "history": history
            }, f)
            
        return ranks, gif_path, json_path
    
    
if __name__ == "__main__":
    # testing
    runner = MatchRunner()
    agent_paths = [
        # "submissions/team_alpha/20260326_120000/agent.py",
        # "submissions/team_beta/20260326_120000/agent.py",
        "agent/smarter_rule_agent.py",
        "agent/genius_rule_agent.py",
        "agent/tactical_rule_agent.py",
        "agent/random_agent.py"
    ]
    team_ids = ["SmarterRuleAgent", "GeniusRuleAgent", "TacticalRuleAgent", "RandomAgent"]
    ranks, gif_path, json_path = runner.run_match(agent_paths, team_ids, seed=45)