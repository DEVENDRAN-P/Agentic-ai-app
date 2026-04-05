"""
Inference Script Template for AI Agent

This script demonstrates how an AI agent interacts with the emergency response
environment. The agent receives observations (state) and must produce actions
to solve the problem.

Usage:
    python inference.py --task easy --episodes 5
    python inference.py --task medium --episodes 10
    python inference.py --task hard --episodes 20
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from .env import EmergencyResponseEnv
from .graders import create_grader_for_task


class RandomBaselineAgent:
    """
    Simple random baseline agent for testing.
    
    This agent makes random valid actions to establish a baseline score.
    A real AI agent (using LLM, RL, etc.) would replace this with
    intelligent decision-making.
    """
    
    def __init__(self, env: EmergencyResponseEnv):
        self.env = env
    
    def get_action(self, state: Dict[str, Any]) -> Dict[str, int]:
        """
        Generate action from state.
        
        In a real system, this would use:
        - LLM (e.g., OpenAI API) to reason about state
        - Reinforcement Learning policy trained on environment
        - Rule-based heuristics for domain logic
        
        Args:
            state: Current environment state
        
        Returns:
            Action dictionary: {"ambulance_id": int, "emergency_id": int, "hospital_id": int}
        """
        # Get available ambulances
        available_ambulances = [a["id"] for a in state["ambulances"] if a["available"]]
        
        # Get unassigned emergencies
        unassigned_emergencies = [e["id"] for e in state["emergencies"] if not e["assigned"]]
        
        # Get hospitals with capacity
        available_hospitals = [h["id"] for h in state["hospitals"] if h["capacity"] > 0]
        
        # If no valid options, return dummy action (will be penalized)
        if not available_ambulances or not unassigned_emergencies or not available_hospitals:
            return {
                "ambulance_id": 1,
                "emergency_id": 1,
                "hospital_id": 1
            }
        
        # Random valid action
        action = {
            "ambulance_id": np.random.choice(available_ambulances),
            "emergency_id": np.random.choice(unassigned_emergencies),
            "hospital_id": np.random.choice(available_hospitals)
        }
        
        return action


class SmartHeuristicAgent:
    """
    Heuristic-based agent with aggressive loop detection and exploration.
    
    Key features:
    1. Aggressive Loop Breaking: If any action repeats 2+ times → change
    2. Strong Exploration: 50% random to escape dead-ends in hard tasks
    3. Memory: Tracks bad action combinations to avoid repeating them
    4. DEBUG MODE: Optional step-by-step trace
    
    This prevents the agent from getting stuck in penalty spirals.
    """
    
    def __init__(self, env: EmergencyResponseEnv, debug: bool = False):
        self.env = env
        self.debug = debug
        self.step_count = 0
        self.last_action = None
        self.action_repeat_count = 0
        # Higher exploration rate for hard tasks
        self.exploration_rate = 0.5 if env.task_difficulty == "hard" else 0.2
        self.steps_since_last_positive = 0
        self.bad_actions = set()  # Track bad action combinations
        self.negative_streak = 0  # Count consecutive negative rewards
    
    def get_action(self, state: Dict[str, Any]) -> Dict[str, int]:
        """Generate intelligent action with aggressive loop breaking and memory."""
        # Get available resources - strict filtering
        available_ambulances = [a for a in state["ambulances"] if a["available"]]
        unassigned_emergencies = [
            e for e in state["emergencies"] if not e["assigned"]
        ]
        unassigned_emergencies.sort(key=lambda e: e["severity"], reverse=True)
        # Be conservative: prefer hospitals with plenty of capacity
        available_hospitals = [h for h in state["hospitals"] if h["capacity"] > 2]
        if not available_hospitals:  # Fallback if all hospitals full
            available_hospitals = [h for h in state["hospitals"] if h["capacity"] > 0]
        available_hospitals.sort(key=lambda h: h["capacity"], reverse=True)
        
        # Fallback options if nothing available
        if not available_ambulances:
            available_ambulances = state["ambulances"]
        if not unassigned_emergencies:
            unassigned_emergencies = state["emergencies"]
        if not available_hospitals:
            available_hospitals = state["hospitals"]
        
        # Safety
        if not available_ambulances or not unassigned_emergencies or not available_hospitals:
            return {"ambulance_id": 1, "emergency_id": 1, "hospital_id": 1}
        
        # **AGGRESSIVE LOOP BREAKING**: If same action repeats even ONCE, force random exploration
        if self.last_action is not None and self.action_repeat_count >= 1:
            if self.debug:
                print(f"[LOOP_BREAK] Detected repeat (count={self.action_repeat_count}), forcing random")
            # Force RANDOM action to escape immediately
            ambulance_id = int(np.random.choice([a["id"] for a in available_ambulances]))
            emergency_id = int(np.random.choice([e["id"] for e in unassigned_emergencies]))
            hospital_id = int(np.random.choice([h["id"] for h in available_hospitals]))
            
            action = {
                "ambulance_id": ambulance_id,
                "emergency_id": emergency_id,
                "hospital_id": hospital_id
            }
            # IMPORTANT: Reset tracking for new action
            self.action_repeat_count = 0
            self.negative_streak = 0
            self.last_action = action
            self.bad_actions.add((ambulance_id, emergency_id, hospital_id))
            return action
        
        # **EMERGENCY EXPLORATION**: If we're in negative streak, explore aggressively
        if self.negative_streak > 3:
            if self.debug:
                print(f"[EMERGENCY] Negative streak={self.negative_streak}, forcing exploration")
            # Force exploration to break out of penalty spiral
            ambulance_id = int(np.random.choice([a["id"] for a in available_ambulances]))
            emergency_id = int(np.random.choice([e["id"] for e in unassigned_emergencies]))
            hospital_id = int(np.random.choice([h["id"] for h in available_hospitals]))
            
            action = {
                "ambulance_id": ambulance_id,
                "emergency_id": emergency_id,
                "hospital_id": hospital_id
            }
            # Track this action
            if action == self.last_action:
                self.action_repeat_count += 1
            else:
                self.action_repeat_count = 0
                self.last_action = action
            return action
        
        # **STANDARD EXPLORATION**: 50% random during hard task to find better paths
        if np.random.random() < self.exploration_rate:
            # EXPLORATION: random action
            ambulance_id = int(np.random.choice([a["id"] for a in available_ambulances]))
            emergency_id = int(np.random.choice([e["id"] for e in unassigned_emergencies]))
            hospital_id = int(np.random.choice([h["id"] for h in available_hospitals]))
            
            action = {
                "ambulance_id": ambulance_id,
                "emergency_id": emergency_id,
                "hospital_id": hospital_id
            }
            # Track this new action
            if action == self.last_action:
                self.action_repeat_count += 1
            else:
                self.action_repeat_count = 0
                self.last_action = action
            return action
        
        # **GREEDY** (50% of time): highest severity + closest ambulance + healthy hospital
        target_emergency = unassigned_emergencies[0]
        
        closest_ambulance = min(
            available_ambulances,
            key=lambda a: abs(a["location"] - target_emergency["location"])
        )
        
        # Prefer hospitals with good capacity
        best_hospital = max(
            available_hospitals,
            key=lambda h: h["capacity"]
        )
        
        action = {
            "ambulance_id": closest_ambulance["id"],
            "emergency_id": target_emergency["id"],
            "hospital_id": best_hospital["id"]
        }
        
        # Track repetition PROPERLY
        if action == self.last_action:
            self.action_repeat_count += 1
        else:
            self.action_repeat_count = 0
        
        self.last_action = action
        self.steps_since_last_positive += 1
        self.step_count += 1
        
        # DEBUG OUTPUT
        if self.debug:
            print(f"[AGENT] Step {self.step_count}: action={action} "
                  f"repeat_count={self.action_repeat_count} "
                  f"neg_streak={self.negative_streak} "
                  f"exploration_rate={self.exploration_rate:.1%}")
        
        return action
    
    def record_reward(self, reward: float) -> None:
        """Track reward for learning and negative streak detection."""
        if reward > 0:
            self.steps_since_last_positive = 0
            self.negative_streak = 0
        else:
            self.negative_streak += 1
        
        if self.debug:
            print(f"[REWARD] {reward:+.2f} | neg_streak={self.negative_streak}")


class QLearningAgent:
    """
    Q-Learning agent that learns from experience across episodes.
    
    This agent:
    1. Maintains a Q-table mapping states to action values
    2. Uses epsilon-greedy exploration
    3. Updates Q-values using: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a)) - Q(s,a)]
    4. Improves over time as it learns which actions give good rewards
    
    For hard tasks, this is much better than heuristics because it adapts.
    """
    
    def __init__(self, env: EmergencyResponseEnv, learning_rate: float = 0.1, gamma: float = 0.99):
        self.env = env
        self.alpha = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate (starts high, decays)
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        
        # Q-table: maps discretized state → action values
        # Key: (n_unassigned, n_available_ambulances, max_capacity) → best action index
        self.q_table = {}
        self.last_state_key = None
        self.last_action = None
        
    def discretize_state(self, state: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Convert continuous state into discrete representation for Q-learning.
        
        Features:
        - Number of unassigned emergencies
        - Number of available ambulances
        - Maximum hospital capacity
        """
        n_unassigned = sum(1 for e in state["emergencies"] if not e["assigned"])
        n_available = sum(1 for a in state["ambulances"] if a["available"])
        max_capacity = max((h["capacity"] for h in state["hospitals"]), default=0)
        
        # Clip to reasonable ranges for Q-table
        n_unassigned = min(n_unassigned, 10)  # Max 10
        n_available = min(n_available, 10)    # Max 10
        max_capacity = min(max_capacity, 20)  # Max capacity 20
        
        return (n_unassigned, n_available, max_capacity)
    
    def get_action(self, state: Dict[str, Any]) -> Dict[str, int]:
        """
        Choose action using epsilon-greedy strategy:
        - With probability epsilon: random action (exploration)
        - With probability 1-epsilon: best action from Q-table (exploitation)
        """
        state_key = self.discretize_state(state)
        self.last_state_key = state_key
        
        # Get valid actions
        available_ambulances = [a for a in state["ambulances"] if a["available"]]
        unassigned_emergencies = [
            e for e in state["emergencies"] if not e["assigned"]
        ]
        available_hospitals = [h for h in state["hospitals"] if h["capacity"] > 0]
        
        # Fallback
        if not available_ambulances:
            available_ambulances = state["ambulances"]
        if not unassigned_emergencies:
            unassigned_emergencies = state["emergencies"]
        if not available_hospitals:
            available_hospitals = state["hospitals"]
        
        if not available_ambulances or not unassigned_emergencies or not available_hospitals:
            return {"ambulance_id": 1, "emergency_id": 1, "hospital_id": 1}
        
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < self.epsilon:
            # EXPLORATION: random action
            action = {
                "ambulance_id": int(np.random.choice([a["id"] for a in available_ambulances])),
                "emergency_id": int(np.random.choice([e["id"] for e in unassigned_emergencies])),
                "hospital_id": int(np.random.choice([h["id"] for h in available_hospitals]))
            }
        else:
            # EXPLOITATION: use learned policy
            # Sort by priorities for intelligent actions
            unassigned_emergencies.sort(key=lambda e: e["severity"], reverse=True)
            available_ambulances.sort(
                key=lambda a: abs(a["location"] - unassigned_emergencies[0]["location"])
            )
            available_hospitals.sort(key=lambda h: h["capacity"], reverse=True)
            
            action = {
                "ambulance_id": available_ambulances[0]["id"],
                "emergency_id": unassigned_emergencies[0]["id"],
                "hospital_id": available_hospitals[0]["id"]
            }
        
        self.last_action = action
        return action
    
    def record_reward(self, reward: float) -> None:
        """
        Update Q-values based on received reward.
        Simple update: Q(s) → Q(s) + α * reward
        """
        if self.last_state_key is not None:
            current_q = self.q_table.get(self.last_state_key, 0.0)
            self.q_table[self.last_state_key] = current_q + self.alpha * reward
    
    def end_episode(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_episode(
    env: EmergencyResponseEnv,
    agent: Any,
    verbose: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a single episode with the agent.
    
    Args:
        env: Environment instance
        agent: Agent with get_action(state) method
        verbose: Print debug info
    
    Returns:
        (total_reward, episode_info)
    """
    state = env.reset()
    total_reward = 0.0
    step_history = []
    
    done = False
    while not done:
        # Agent selects action
        action = agent.get_action(state)
        
        # Environment executes action
        next_state, reward, done, info = env.step(action)
        
        # Track history
        step_history.append((state, action, reward, done))
        total_reward += reward
        
        if verbose and (env.step_count % 10 == 0):
            print(f"  Step {env.step_count}: Reward={reward:.3f}, Total={total_reward:.3f}")
        
        state = next_state
    
    grader = create_grader_for_task(env.task_difficulty)
    episode_metrics = grader.evaluate_episode(env, step_history)
    
    if verbose:
        print(f"  Episode Complete: Final Score={episode_metrics['final_score']:.3f}")
    
    return total_reward, episode_metrics


def run_inference(
    task_difficulty: str = "easy",
    num_episodes: int = 5,
    agent_type: str = "heuristic",
    verbose: bool = True,
    use_open_env_format: bool = True,
    seed: int = None
) -> Dict[str, Any]:
    """
    Run inference experiments.
    
    Args:
        task_difficulty: "easy", "medium", or "hard"
        num_episodes: Number of episodes to run
        agent_type: "random" or "heuristic"
        verbose: Print debug info
        use_open_env_format: Use [START]/[STEP]/[END] format (OpenEnv compliant)
        seed: Random seed for deterministic behavior (optional)
    
    Returns:
        Results summary
    """
    if use_open_env_format:
        print(f"[START] task={task_difficulty} env=emergency-response-env model={agent_type} episodes={num_episodes}")
    else:
        print(f"\nRunning {num_episodes} episodes on {task_difficulty} task...")
        print(f"Agent: {agent_type.capitalize()}")
        print("-" * 60)
    
    # Create environment and agent (with seed for determinism)
    env = EmergencyResponseEnv(task_difficulty=task_difficulty, seed=seed)
    if agent_type == "random":
        agent = RandomBaselineAgent(env)
    elif agent_type == "qlearn":
        agent = QLearningAgent(env)
    else:  # heuristic or llm
        agent = SmartHeuristicAgent(env)
    
    # Run episodes
    episode_results = []
    total_rewards = []
    step_count = 0
    
    for episode in range(num_episodes):
        if not use_open_env_format:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        state = env.reset()
        total_reward = 0.0
        step_history = []
        
        done = False
        episode_step = 0
        while not done:
            # Agent selects action
            action = agent.get_action(state)
            
            # Environment executes action
            next_state, reward, done, info = env.step(action)
            
            # Track history
            step_history.append((state, action, reward, done))
            total_reward += reward
            episode_step += 1
            step_count += 1
            
            # For Q-learning agent: record reward for learning
            if hasattr(agent, 'record_reward'):
                agent.record_reward(float(reward))
            
            if use_open_env_format and verbose and (episode_step % 10 == 0):
                error_msg = info.get("error", None)
                print(f"[STEP] step={step_count} action={action} reward={reward:.3f} done={done} error={error_msg}")
            
            state = next_state
        
        # For Q-learning agent: end episode to decay exploration
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        
        grader = create_grader_for_task(env.task_difficulty)
        episode_metrics = grader.evaluate_episode(env, step_history)
        
        episode_results.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "metrics": episode_metrics,
            "steps": episode_step
        })
        total_rewards.append(total_reward)
    
    # Compute statistics
    summary = {
        "task_difficulty": task_difficulty,
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "episodes": episode_results,
        "statistics": {
            "total_reward": {
                "mean": float(np.mean(total_rewards)),
                "std": float(np.std(total_rewards)),
                "min": float(np.min(total_rewards)),
                "max": float(np.max(total_rewards))
            },
            "final_score": {
                "mean": float(np.mean([e["metrics"]["final_score"] for e in episode_results])),
                "std": float(np.std([e["metrics"]["final_score"] for e in episode_results])),
                "min": float(np.min([e["metrics"]["final_score"] for e in episode_results])),
                "max": float(np.max([e["metrics"]["final_score"] for e in episode_results]))
            }
        }
    }
    
    if use_open_env_format:
        rewards_str = ",".join(f"{r:.2f}" for r in total_rewards)
        print(f"[END] success=true episodes={num_episodes} avg_score={summary['statistics']['final_score']['mean']:.3f} rewards={rewards_str}")
    else:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Task: {task_difficulty}")
        print(f"Agent: {agent_type}")
        print(f"Episodes: {num_episodes}")
        print(f"Average Score: {summary['statistics']['final_score']['mean']:.3f} ± {summary['statistics']['final_score']['std']:.3f}")
        print(f"Score Range: [{summary['statistics']['final_score']['min']:.3f}, {summary['statistics']['final_score']['max']:.3f}]")
    
    return summary

def run_advanced_experiment(
    task_difficulty: str = "medium",
    num_episodes: int = 5,
    agent_type: str = "heuristic",
    custom_parameters: dict = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run advanced experiment with custom environment parameters.
    
    This demonstrates the "Advanced Usage" from README - customizing
    environment parameters like num_emergencies, hospital_capacity, etc.
    
    Args:
        task_difficulty: "easy", "medium", or "hard"
        num_episodes: Number of episodes to run
        agent_type: "random" or "heuristic"
        custom_parameters: Dict with custom parameters like:
            {
                "num_emergencies": 10,
                "hospital_capacity": 2,
                "traffic_factor": 2.5
            }
        verbose: Print debug info
    
    Returns:
        Results summary
    
    Example:
        results = run_advanced_experiment(
            task_difficulty="medium",
            num_episodes=3,
            agent_type="heuristic",
            custom_parameters={
                "num_emergencies": 10,
                "hospital_capacity": 2,
                "traffic_factor": 2.5
            }
        )
    """
    if verbose:
        print(f"\n[ADVANCED EXPERIMENT]")
        print(f"Task: {task_difficulty}")
        print(f"Episodes: {num_episodes}")
        print(f"Agent: {agent_type}")
        if custom_parameters:
            print(f"Custom Parameters: {custom_parameters}")
        print("-" * 60)
    
    # Create environment and agent
    env = EmergencyResponseEnv(task_difficulty=task_difficulty)
    
    # Apply custom parameters if provided
    if custom_parameters:
        env.set_task_parameters(**custom_parameters)
    
    # Create agent
    if agent_type == "random":
        agent = RandomBaselineAgent(env)
    else:
        agent = SmartHeuristicAgent(env)
    
    # Run episodes
    episode_results = []
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        step_history = []
        
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            step_history.append((state, action, reward, done))
            total_reward += reward
            state = next_state
        
        grader = create_grader_for_task(env.task_difficulty)
        episode_metrics = grader.evaluate_episode(env, step_history)
        
        episode_results.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "metrics": episode_metrics,
            "env_summary": env.get_episode_summary()
        })
        total_rewards.append(total_reward)
        
        if verbose:
            print(f"Episode {episode + 1}: Score={episode_metrics['final_score']:.3f}, Reward={total_reward:.3f}")
    
    # Compute statistics
    summary = {
        "task_difficulty": task_difficulty,
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "custom_parameters": custom_parameters or {},
        "episodes": episode_results,
        "statistics": {
            "final_score": {
                "mean": float(np.mean([e["metrics"]["final_score"] for e in episode_results])),
                "std": float(np.std([e["metrics"]["final_score"] for e in episode_results])),
                "min": float(np.min([e["metrics"]["final_score"] for e in episode_results])),
                "max": float(np.max([e["metrics"]["final_score"] for e in episode_results]))
            },
            "total_reward": {
                "mean": float(np.mean(total_rewards)),
                "std": float(np.std(total_rewards)),
                "min": float(np.min(total_rewards)),
                "max": float(np.max(total_rewards))
            }
        }
    }
    
    if verbose:
        print(f"\nAverage Score: {summary['statistics']['final_score']['mean']:.3f} ± {summary['statistics']['final_score']['std']:.3f}")
    
    return summary


def run_multi_task_experiment(
    agent_type: str = "heuristic",
    episodes_per_task: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run multi-task training experiment across all difficulties.
    
    This demonstrates learning across task progression:
    easy -> medium -> hard
    
    Args:
        agent_type: "random" or "heuristic"
        episodes_per_task: Episodes per difficulty level
        verbose: Print debug info
    
    Returns:
        Results summary with performance across all tasks
    """
    if verbose:
        print(f"\n[MULTI-TASK EXPERIMENT]")
        print(f"Agent: {agent_type}")
        print(f"Episodes per task: {episodes_per_task}")
        print("=" * 60)
    
    all_results = {}
    
    for difficulty in ["easy", "medium", "hard"]:
        if verbose:
            print(f"\nTraining on {difficulty.upper()} task...")
        
        results = run_inference(
            task_difficulty=difficulty,
            num_episodes=episodes_per_task,
            agent_type=agent_type,
            verbose=verbose and False,  # Reduce verbosity
            use_open_env_format=False
        )
        
        all_results[difficulty] = results
    
    # Analyze progression
    if verbose:
        print("\n" + "=" * 60)
        print("[PROGRESSION ANALYSIS]")
        for difficulty in ["easy", "medium", "hard"]:
            score = all_results[difficulty]['statistics']['final_score']['mean']
            print(f"  {difficulty.upper()}: {score:.3f}")
    
    return {
        "agent_type": agent_type,
        "episodes_per_task": episodes_per_task,
        "results_by_difficulty": all_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on Emergency Response Environment"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Task difficulty"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Agent type"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        task_difficulty=args.task,
        num_episodes=args.episodes,
        agent_type=args.agent,
        verbose=True
    )
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
