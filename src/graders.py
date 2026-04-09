"""
Grader Design for Emergency Response Environment

Scoring System aligned with hackathon criteria:
- Metric 1: Average response time (lower is better)
- Metric 2: % high-severity cases handled first (higher is better)
- Metric 3: Resource utilization efficiency (higher is better)

Final Score Calculation:
score = 0.5*(priority_handling) + 0.3*(response_speed) + 0.2*(resource_usage)

All metrics are normalized to [0.0, 1.0] for consistent evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from .env import EmergencyResponseEnv


class EmergencyResponseGrader:
    """
    Grader for evaluating AI agent performance on emergency response tasks.
    
    Evaluation Metrics:
    1. Priority Handling (50% weight): % of high-severity cases handled appropriately
    2. Response Speed (30% weight): Average response time efficiency
    3. Resource Usage (20% weight): Ambulance and hospital utilization
    """
    
    def __init__(self):
        self.metrics = {
            "priority_handling": 0.0,
            "response_speed": 0.0,
            "resource_usage": 0.0,
            "final_score": 0.0
        }
        self.episode_stats = []
    
    def evaluate_episode(self, env: EmergencyResponseEnv, step_history: List[Tuple]) -> Dict[str, float]:
        """
        Evaluate a complete episode.
        
        Args:
            env: Environment instance after episode completion
            step_history: List of (state, action, reward, done) tuples from episode
        
        Returns:
            Dictionary with all metrics and final score
        """
        stats = {
            "priority_handling": self._calculate_priority_handling(env, step_history),
            "response_speed": self._calculate_response_speed(env, step_history),
            "resource_usage": self._calculate_resource_usage(env, step_history),
        }
        
        # Calculate step reward quality as a penalty factor
        # If agent gets many negative rewards, score should reflect it
        reward_quality = self._calculate_reward_quality(step_history)
        
        # Calculate final score (weighted combination)
        stats["final_score"] = (
            0.5 * stats["priority_handling"] +
            0.3 * stats["response_speed"] +
            0.2 * stats["resource_usage"]
        )
        
        # APPLY REWARD QUALITY PENALTY
        # If agent got many negative rewards, penalize the score
        stats["final_score"] *= reward_quality
        
        # Clamp to (0, 1) - STRICTLY between 0 and 1 per hackathon rules
        stats["final_score"] = max(0.01, min(1.0 - 0.01, stats["final_score"]))
        
        # Track reward quality for debugging
        stats["reward_quality"] = reward_quality
        
        self.episode_stats.append(stats)
        return stats
    
    def _calculate_priority_handling(self, env: EmergencyResponseEnv, step_history: List) -> float:
        """
        Calculate priority handling score.
        
        Metric: % of high-severity (severity >= 8) emergencies handled appropriately
        
        Returns:
            Normalized score [0.0, 1.0]
        """
        if env.total_high_severity == 0:
            return 1.0 - 0.01  # Perfect score if no high-severity cases
        
        # Calculate percentage of high-severity handled
        high_severity_percentage = env.high_severity_handled / max(env.total_high_severity, 1)
        
        # Apply penalty for delayed handling
        # Each step of delay reduces score
        penalty = 0.0
        for emergency in env.emergencies:
            if emergency["severity"] >= 8:
                if emergency["time_waiting"] > 5:  # Allowed 5 steps of waiting
                    penalty += (emergency["time_waiting"] - 5) * 0.01
        
        priority_score = high_severity_percentage - min(penalty, 0.3)
        return max(0.01, min(1.0 - 0.01, priority_score))
    
    def _calculate_response_speed(self, env: EmergencyResponseEnv, step_history: List) -> float:
        """
        Calculate response speed score.
        
        Metric: Average response time efficiency
        - Target: Keep average response time low
        - Formula: 1 - (actual_avg_time / max_acceptable_time)
        
        Returns:
            Normalized score [0.0, 1.0]
        """
        num_assigned = sum(1 for e in env.emergencies if e["assigned_ambulance"] is not None)
        
        if num_assigned == 0:
            return 0.01  # No emergencies handled, return a score just above 0
        
        average_response_time = env.total_response_time / num_assigned
        
        # Define max acceptable time based on difficulty
        if env.task_difficulty == "easy":
            max_acceptable_time = 20.0
        elif env.task_difficulty == "medium":
            max_acceptable_time = 30.0
        else:  # hard
            max_acceptable_time = 40.0
        
        # Calculate speed score (upper-bounded at 1.0)
        speed_score = 1.0 - (average_response_time / max_acceptable_time)
        return max(0.01, min(1.0 - 0.01, speed_score))
    
    def _calculate_resource_usage(self, env: EmergencyResponseEnv, step_history: List) -> float:
        """
        Calculate resource utilization efficiency.
        
        Metrics:
        1. Ambulance utilization: % of time ambulances were assigned
        2. Hospital capacity efficiency: Avoid overloading single hospitals
        
        Returns:
            Normalized score [0.0, 1.0]
        """
        # Calculate ambulance utilization
        num_assigned_emergencies = sum(1 for e in env.emergencies if e["assigned_ambulance"] is not None)
        ambulance_utilization = num_assigned_emergencies / env.num_ambulances if env.num_ambulances > 0 else 0
        
        # Calculate hospital capacity efficiency (balance across hospitals)
        hospital_loads = [h["patients"] for h in env.hospitals]
        
        if len(hospital_loads) > 0:
            avg_load = np.mean(hospital_loads)
            load_variance = np.var(hospital_loads)
            
            # Prefer balanced utilization (low variance)
            max_acceptable_variance = (env.hospital_capacity ** 2) / 4
            balance_score = 1.0 - (load_variance / max(max_acceptable_variance, 1.0))
            balance_score = max(0.0, min(1.0, balance_score))
        else:
            balance_score = 1.0
        
        # Combined resource score
        resource_score = 0.6 * ambulance_utilization + 0.4 * balance_score
        return max(0.01, min(1.0 - 0.01, resource_score))
    
    def _calculate_reward_quality(self, step_history: List[Tuple]) -> float:
        """
        Calculate reward quality penalty.
        
        If agent gets many invalid actions with negative rewards,
        the episode score should be penalized accordingly.
        
        Returns:
            Quality multiplier in [0.0, 1.0]
            - 1.0 = good quality (mostly positive rewards)
            - 0.5 = poor quality (many negative rewards)
            - 0.0 = very poor (mostly negative)
        
        This prevents inflated scores when agent makes many invalid actions.
        """
        if not step_history:
            return 1.0
        
        # Extract rewards from step history (3rd element of each tuple)
        rewards = [float(step[2]) for step in step_history]
        
        # Count negative rewards
        negative_count = sum(1 for r in rewards if r < 0)
        total_steps = len(rewards)
        
        negative_rate = negative_count / total_steps if total_steps > 0 else 0
        
        # If more than 50% of steps are negative, apply penalty
        # REDUCED penalty from 0.7 to 0.5 - less harsh
        reward_quality = 1.0 - (negative_rate * 0.5)
        
        return max(0.5, min(1.0, reward_quality))
    
    def calculate_emergencies_handled(self, env: EmergencyResponseEnv) -> float:
        """
        Calculate percentage of emergencies that were handled.
        
        Returns:
            Fraction in [0.0, 1.0]
            - 1.0 = all emergencies handled
            - 0.6 = 60% of emergencies handled (success threshold)
            - 0.0 = no emergencies handled
        """
        total_emergencies = len(env.emergencies)
        handled = sum(1 for e in env.emergencies if e["assigned_ambulance"] is not None)
        
        return handled / total_emergencies if total_emergencies > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluated episodes."""
        if not self.episode_stats:
            return {"message": "No episodes evaluated yet"}
        
        stats_array = np.array([
            [s["priority_handling"], s["response_speed"], s["resource_usage"], s["final_score"]]
            for s in self.episode_stats
        ])
        
        return {
            "episodes_evaluated": len(self.episode_stats),
            "priority_handling": {
                "mean": float(np.mean(stats_array[:, 0])),
                "std": float(np.std(stats_array[:, 0])),
                "min": float(np.min(stats_array[:, 0])),
                "max": float(np.max(stats_array[:, 0]))
            },
            "response_speed": {
                "mean": float(np.mean(stats_array[:, 1])),
                "std": float(np.std(stats_array[:, 1])),
                "min": float(np.min(stats_array[:, 1])),
                "max": float(np.max(stats_array[:, 1]))
            },
            "resource_usage": {
                "mean": float(np.mean(stats_array[:, 2])),
                "std": float(np.std(stats_array[:, 2])),
                "min": float(np.min(stats_array[:, 2])),
                "max": float(np.max(stats_array[:, 2]))
            },
            "final_score": {
                "mean": float(np.mean(stats_array[:, 3])),
                "std": float(np.std(stats_array[:, 3])),
                "min": float(np.min(stats_array[:, 3])),
                "max": float(np.max(stats_array[:, 3]))
            }
        }


# Task-specific graders for difficulty progression

class EasyTaskGrader(EmergencyResponseGrader):
    """Grader for easy task: basic ambulance-to-emergency assignment."""
    
    def evaluate_episode(self, env: EmergencyResponseEnv, step_history: List[Tuple]) -> Dict[str, float]:
        """Easy task focuses on making any valid assignment."""
        stats = super().evaluate_episode(env, step_history)
        
        # Bonus for assigning any emergencies quickly
        num_assigned = sum(1 for e in env.emergencies if e["assigned_ambulance"] is not None)
        assignment_bonus = (num_assigned / len(env.emergencies)) * 0.1
        
        stats["final_score"] = min(1.0 - 0.01, stats["final_score"] + assignment_bonus)
        return stats


class MediumTaskGrader(EmergencyResponseGrader):
    """Grader for medium task: prioritization under constraints."""
    
    def evaluate_episode(self, env: EmergencyResponseEnv, step_history: List[Tuple]) -> Dict[str, float]:
        """Medium task emphasizes priority handling."""
        stats = super().evaluate_episode(env, step_history)
        
        # Increase weight on priority handling for medium difficulty
        stats["final_score"] = (
            0.6 * stats["priority_handling"] +  # Increased from 0.5
            0.25 * stats["response_speed"] +     # Decreased from 0.3
            0.15 * stats["resource_usage"]       # Decreased from 0.2
        )
        
        stats["final_score"] = max(0.01, min(1.0 - 0.01, stats["final_score"]))
        return stats


class HardTaskGrader(EmergencyResponseGrader):
    """Grader for hard task: complex decision-making under high stress."""
    
    def evaluate_episode(self, env: EmergencyResponseEnv, step_history: List[Tuple]) -> Dict[str, float]:
        """Hard task requires balanced optimization across all metrics."""
        stats = super().evaluate_episode(env, step_history)
        
        # Apply stricter scoring thresholds for hard task
        penalty = 0.0
        
        # Penalty for unhandled emergencies
        unhandled = sum(1 for e in env.emergencies if e["assigned_ambulance"] is None)
        penalty += (unhandled / len(env.emergencies)) * 0.2
        
        # Penalty for high response times
        if env.step_count > env.max_steps * 0.8:
            penalty += 0.1
        
        stats["final_score"] = max(0.01, stats["final_score"] - penalty)
        return stats


def create_grader_for_task(task_difficulty: str) -> EmergencyResponseGrader:
    """Factory function to create appropriate grader for task difficulty."""
    if task_difficulty == "easy":
        return EasyTaskGrader()
    elif task_difficulty == "medium":
        return MediumTaskGrader()
    else:  # hard
        return HardTaskGrader()
