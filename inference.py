#!/usr/bin/env python3
"""
OpenEnv Inference Script - Emergency Response Environment

STRICT LOGGING FORMAT (REQUIRED):
  [START] task=<task> env=emergency model=<model> episodes=<n>
  [STEP] episode=<n> step=<n> action=... reward=<0.00>
  [END] success=true episodes=<n> avg_score=<0.000> rewards=<r1,r2,...>

Features:
- OpenAI API integration with environment variables
- Deterministic grading (same input → same output)
- Seed-based reproducibility
- Full step-by-step logging
"""

import os
import sys
import json
import random
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.env import EmergencyResponseEnv
from src.graders import create_grader_for_task
from src.inference import RandomBaselineAgent, SmartHeuristicAgent

# Environment variables for OpenAI
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def log_start(task: str, env: str, model: str, episodes: int) -> None:
    """Emit [START] log - EXACT FORMAT"""
    print(f"[START] task={task} env={env} model={model} episodes={episodes}", flush=True)


def log_step(episode: int, step: int, action: str, reward: float) -> None:
    """Emit [STEP] log - EXACT FORMAT (includes episode number)"""
    print(f"[STEP] episode={episode} step={step} action={action} reward={reward:.3f}", flush=True)


def log_end(success: bool, episodes: int, avg_score: float, rewards: List[float]) -> None:
    """Emit [END] log - EXACT FORMAT"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} episodes={episodes} avg_score={avg_score:.3f} rewards={rewards_str}", flush=True)



class OpenAIAgent:
    """Agent that uses OpenAI API for decision-making."""
    
    def __init__(self, env: EmergencyResponseEnv):
        self.env = env
        self.client = None
        
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=API_BASE_URL if API_BASE_URL != "https://api.openai.com/v1" else None
                )
            except Exception as e:
                print(f"Warning: OpenAI client failed: {e}. Using heuristic fallback.", file=sys.stderr)
        
        # Fallback agent
        self.fallback_agent = SmartHeuristicAgent(env)
    
    def state_to_prompt(self, state: Dict[str, Any]) -> str:
        """Convert state to natural language prompt."""
        return f"""Emergency Response Coordination:
- {len(state['emergencies'])} emergencies active
- {sum(1 for a in state['ambulances'] if a['available'])}/{len(state['ambulances'])} ambulances available
- {sum(h['capacity'] for h in state['hospitals'])} hospital beds available
- Traffic level: {state['traffic_level']}/5

Return only 3 numbers (ambulance_id, emergency_id, hospital_id) without explanation."""
    
    def get_action(self, state: Dict[str, Any]) -> Dict[str, int]:
        """Get action from OpenAI or fallback."""
        if not self.client:
            return self.fallback_agent.get_action(state)
        
        try:
            prompt = self.state_to_prompt(state)
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            
            # Parse response
            import re
            numbers = re.findall(r'\d+', response.choices[0].message.content)
            if len(numbers) >= 3:
                return {
                    "ambulance_id": min(int(numbers[0]), self.env.num_ambulances),
                    "emergency_id": min(int(numbers[1]), len(self.env.emergencies)),
                    "hospital_id": min(int(numbers[2]), self.env.num_hospitals)
                }
        except:
            pass
        
        return self.fallback_agent.get_action(state)


def run_inference(
    task_difficulty: str = "easy",
    num_episodes: int = 5,
    agent_type: str = "heuristic",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run inference with strict logging format.
    
    Args:
        task_difficulty: "easy", "medium", "hard"
        num_episodes: Number of episodes
        agent_type: "random", "heuristic", "llm"
        seed: Random seed for reproducibility
    """
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Log start
    model_name = MODEL_NAME if agent_type == "llm" else agent_type
    log_start(task=task_difficulty, env="emergency-response-env", model=model_name, episodes=num_episodes)
    
    # Create environment
    env = EmergencyResponseEnv(task_difficulty=task_difficulty)
    
    # Create agent
    if agent_type == "llm":
        agent = OpenAIAgent(env)
    elif agent_type == "random":
        agent = RandomBaselineAgent(env)
    else:  # heuristic
        agent = SmartHeuristicAgent(env)
    
    # Run episodes
    episode_results = []
    episode_rewards = []
    all_rewards_flat = []
    global_step = 0
    
    for episode_num in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        step_history = []
        episode_step = 0
        done = False
        
        while not done:
            episode_step += 1
            global_step += 1
            
            # Agent decides
            action = agent.get_action(state)
            
            # Environment executes
            next_state, reward, done, info = env.step(action)
            step_history.append((state, action, reward, done))
            episode_reward += reward
            all_rewards_flat.append(reward)
            
            # Log step (MUST include episode number)
            action_str = f"({action.get('ambulance_id', 0)},{action.get('emergency_id', 0)},{action.get('hospital_id', 0)})"
            log_step(episode=episode_num, step=episode_step, action=action_str, reward=float(reward))
            
            state = next_state
        
        # Grade episode using deterministic grader
        grader = create_grader_for_task(task_difficulty)
        episode_metrics = grader.evaluate_episode(env, step_history)
        
        episode_results.append({
            "episode": episode_num,
            "reward": float(episode_reward),
            "score": float(episode_metrics.get("final_score", 0.0)),
            "steps": episode_step,
            "metrics": episode_metrics
        })
        
        episode_rewards.append(float(episode_reward))
    
    # Calculate statistics
    final_scores = [e["score"] for e in episode_results]
    avg_score = float(np.mean(final_scores)) if final_scores else 0.0
    
    # Log end (STRICT FORMAT)
    log_end(success=True, episodes=num_episodes, avg_score=avg_score, rewards=episode_rewards)
    
    return {
        "task_difficulty": task_difficulty,
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "episodes": episode_results,
        "statistics": {
            "avg_score": avg_score,
            "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "final_scores": final_scores,
            "rewards": episode_rewards
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenEnv Emergency Response Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy", help="Task difficulty")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--agent", choices=["random", "heuristic", "llm"], default="heuristic", help="Agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    
    args = parser.parse_args()
    
    try:
        results = run_inference(
            task_difficulty=args.task,
            num_episodes=args.episodes,
            agent_type=args.agent,
            seed=args.seed
        )
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
