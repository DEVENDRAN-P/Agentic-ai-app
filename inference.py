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
from src.inference import RandomBaselineAgent, SmartHeuristicAgent, QLearningAgent

# Environment variables for OpenAI (per hackathon requirements)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
OPENAI_API_KEY = HF_TOKEN

# Inference parameters
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] log - EXACT FORMAT per hackathon requirements"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] log - EXACT FORMAT per hackathon requirements"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log - EXACT FORMAT per hackathon requirements"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)



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
    seed: int = 42,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run inference with strict logging format.
    
    Args:
        task_difficulty: "easy", "medium", "hard"
        num_episodes: Number of episodes
        agent_type: "random", "heuristic", "llm"
        seed: Random seed for reproducibility
        debug: Enable debug output
    """
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = EmergencyResponseEnv(task_difficulty=task_difficulty)
    
    # Create agent
    if agent_type == "llm":
        agent = OpenAIAgent(env)
    elif agent_type == "random":
        agent = RandomBaselineAgent(env)
    elif agent_type == "qlearn":
        agent = QLearningAgent(env)
    else:  # heuristic
        agent = SmartHeuristicAgent(env, debug=debug)
    
    # Run episodes
    episode_results = []
    all_episode_rewards = []
    
    # model_name for logging
    model_name = MODEL_NAME if agent_type == "llm" else agent_type
    
    for episode_num in range(1, num_episodes + 1):
        # Log start for THIS episode
        log_start(task=task_difficulty, env="emergency-response-env", model=model_name)
        
        state = env.reset()
        episode_reward = 0.0
        step_history = []
        episode_step = 0
        step_rewards = []  # Per-episode step rewards for logging
        done = False
        stuck_counter = 0  # Track repeated bad actions
        last_action = None
        
        while not done:
            episode_step += 1
            
            # Agent decides
            action = agent.get_action(state)
            
            # Detect stuck behavior - same action repeated with negative reward
            if action == last_action and (episode_step > 1 and step_rewards[-1] < -0.2):
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # Early termination if stuck for too long (more than 3 repeated bad actions)
            if stuck_counter > 3:
                done = True
                reward = 0.0  # No reward for forced termination
            else:
                # Environment executes
                next_state, reward, done, info = env.step(action)
                state = next_state
            
            # Record reward for agent's loop detection (if agent supports it)
            if hasattr(agent, 'record_reward'):
                agent.record_reward(float(reward))
            
            step_history.append((state, action, reward, done))
            episode_reward += reward
            
            # Log step (per hackathon format) - use episode_step, not global
            action_str = f"({action.get('ambulance_id', 0)},{action.get('emergency_id', 0)},{action.get('hospital_id', 0)})"
            log_step(step=episode_step, action=action_str, reward=float(reward), done=done, error=None)
            step_rewards.append(float(reward))
            
            last_action = action
        
        # Grade episode using deterministic grader
        grader = create_grader_for_task(task_difficulty)
        episode_metrics = grader.evaluate_episode(env, step_history)
        episode_score = float(episode_metrics.get("final_score", 0.0))
        
        # For Q-learning agent: end episode to decay exploration
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        
        # Emit [END] for this episode
        success = episode_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=episode_step, score=episode_score, rewards=step_rewards)
        
        episode_results.append({
            "episode": episode_num,
            "reward": float(episode_reward),
            "score": episode_score,
            "steps": episode_step,
            "metrics": episode_metrics
        })
        
        all_episode_rewards.append(float(episode_reward))
    
    # Calculate statistics for summary
    final_scores = [e["score"] for e in episode_results]
    avg_score = float(np.mean(final_scores)) if final_scores else 0.0
    
    return {
        "task_difficulty": task_difficulty,
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "episodes": episode_results,
        "statistics": {
            "avg_score": avg_score,
            "avg_reward": float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0,
            "final_scores": final_scores,
            "episode_rewards": all_episode_rewards
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenEnv Emergency Response Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy", help="Task difficulty")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--agent", choices=["random", "heuristic", "qlearn", "llm"], default="heuristic", help="Agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (shows agent decisions)")
    
    args = parser.parse_args()
    
    try:
        results = run_inference(
            task_difficulty=args.task,
            num_episodes=args.episodes,
            agent_type=args.agent,
            seed=args.seed,
            debug=args.debug
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
