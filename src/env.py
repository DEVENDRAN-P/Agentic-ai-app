"""
Smart Emergency Response Environment - OpenEnv Implementation

This environment simulates emergency response coordination where an AI agent
learns to assign ambulances to emergencies and select hospitals efficiently,
while managing constraints like ambulance availability, hospital capacity,
and traffic delays.

Real-world impact: Optimizes emergency response systems used in smart cities,
disaster management, and healthcare logistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import random


class EmergencyResponseEnv:
    """
    OpenEnv environment for emergency response optimization.
    
    State Space:
    - emergencies: List of active emergencies with severity, location, waiting time
    - ambulances: List of ambulances with location and availability status
    - hospitals: List of hospitals with current capacity
    - traffic_level: Current traffic congestion (1-5 scale)
    
    Action Space:
    - ambulance_id: Which ambulance to dispatch (1 to num_ambulances)
    - emergency_id: Which emergency to respond to (1 to num_emergencies)
    - hospital_id: Which hospital to send patient to (1 to num_hospitals)
    
    Reward Components (Sum to 1.0):
    - priority_handling (0.5): Reward for handling high-severity cases first
    - response_speed (0.3): Reward for quick response time
    - resource_usage (0.2): Reward for efficient ambulance/hospital utilization
    """
    
    def __init__(self, task_difficulty: str = "easy"):
        """
        Initialize emergency response environment.
        
        Args:
            task_difficulty: "easy", "medium", or "hard"
                - easy: 2-3 emergencies, all ambulances available, hospitals free
                - medium: 4-5 emergencies, some ambulances busy, limited capacity
                - hard: 6-8 emergencies, traffic delays, hospital overload
        """
        self.task_difficulty = task_difficulty
        self._init_task_parameters()
        
        # Episode tracking metrics
        self.total_response_time = 0
        self.high_severity_handled = 0
        self.total_high_severity = 0
        self.step_count = 0
        self.max_steps = 100
        
        # Episode statistics (for summary)
        self.episode_history = []
        self.current_episode_stats = {}
        
        # Initialize state
        self.reset()
    
    def _init_task_parameters(self):
        """Set environment parameters based on task difficulty."""
        if self.task_difficulty == "easy":
            self.num_emergencies = 3
            self.initial_busy_ambulances = 0
            self.hospital_capacity = 10
            self.traffic_factor = 1.0
            self.emergency_spawn_rate = 0.1
        elif self.task_difficulty == "medium":
            self.num_emergencies = 5
            self.initial_busy_ambulances = 2
            self.hospital_capacity = 4
            self.traffic_factor = 1.5
            self.emergency_spawn_rate = 0.15
        else:  # hard
            self.num_emergencies = 8
            self.initial_busy_ambulances = 4
            self.hospital_capacity = 2
            self.traffic_factor = 2.0
            self.emergency_spawn_rate = 0.2
        
        self.num_ambulances = 6
        self.num_hospitals = 3
        self.grid_size = 10
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation (state) as dictionary
        """
        self.step_count = 0
        self.total_response_time = 0
        self.high_severity_handled = 0
        self.total_high_severity = 0
        
        # Initialize emergencies
        self.emergencies = []
        for i in range(self.num_emergencies):
            self.emergencies.append({
                "id": i + 1,
                "severity": random.randint(1, 10),
                "location": random.randint(0, self.grid_size - 1),
                "time_waiting": 0,
                "assigned_ambulance": None,
                "assigned_hospital": None
            })
        
        # Initialize ambulances
        self.ambulances = []
        for i in range(self.num_ambulances):
            is_busy = i < self.initial_busy_ambulances
            self.ambulances.append({
                "id": i + 1,
                "location": random.randint(0, self.grid_size - 1),
                "available": not is_busy,
                "busy_until": random.randint(5, 15) if is_busy else 0,
                "current_emergency": None
            })
        
        # Initialize hospitals
        self.hospitals = []
        for i in range(self.num_hospitals):
            self.hospitals.append({
                "id": i + 1,
                "location": random.randint(0, self.grid_size - 1),
                "current_capacity": self.hospital_capacity,
                "max_capacity": self.hospital_capacity,
                "patients": 0
            })
        
        # Set initial traffic level
        self.traffic_level = random.randint(1, 5)
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "emergencies": [
                {
                    "id": e["id"],
                    "severity": e["severity"],
                    "location": e["location"],
                    "time_waiting": e["time_waiting"],
                    "assigned": e["assigned_ambulance"] is not None
                }
                for e in self.emergencies
            ],
            "ambulances": [
                {
                    "id": a["id"],
                    "location": a["location"],
                    "available": a["available"],
                    "busy_until": a["busy_until"]
                }
                for a in self.ambulances
            ],
            "hospitals": [
                {
                    "id": h["id"],
                    "location": h["location"],
                    "capacity": h["current_capacity"],
                    "patients": h["patients"]
                }
                for h in self.hospitals
            ],
            "traffic_level": self.traffic_level,
            "step": self.step_count
        }
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: {"ambulance_id": int, "emergency_id": int, "hospital_id": int}
        
        Returns:
            (state, reward, done, info)
        """
        self.step_count += 1
        reward = 0.0
        
        ambulance_id = action.get("ambulance_id")
        emergency_id = action.get("emergency_id")
        hospital_id = action.get("hospital_id")
        
        # Validate action
        valid_action, validation_info = self._validate_action(ambulance_id, emergency_id, hospital_id)
        
        if valid_action:
            # Execute assignment
            reward = self._execute_assignment(ambulance_id, emergency_id, hospital_id)
        else:
            # Penalty for invalid action
            reward = -0.4
        
        # Update environment state
        self._update_environment()
        
        # Check if episode is done
        done = self.step_count >= self.max_steps or self._all_emergencies_handled()
        
        # Compile info
        info = {
            "valid_action": valid_action,
            "validations": validation_info,
            "total_response_time": self.total_response_time,
            "high_severity_handled": self.high_severity_handled,
            "total_high_severity": self.total_high_severity
        }
        
        return self._get_state(), reward, done, info
    
    def _validate_action(self, ambulance_id: int, emergency_id: int, hospital_id: int) -> Tuple[bool, Dict]:
        """
        Validate if action is legal.
        
        Returns:
            (is_valid, validation_info)
        """
        info = {}
        
        # Check ambulance exists and is available
        if not (1 <= ambulance_id <= self.num_ambulances):
            info["ambulance_error"] = "Invalid ambulance ID"
            return False, info
        
        ambulance = self.ambulances[ambulance_id - 1]
        if not ambulance["available"]:
            info["ambulance_error"] = "Ambulance not available"
            return False, info
        
        # Check emergency exists
        if not (1 <= emergency_id <= len(self.emergencies)):
            info["emergency_error"] = "Invalid emergency ID"
            return False, info
        
        emergency = self.emergencies[emergency_id - 1]
        if emergency["assigned_ambulance"] is not None:
            info["emergency_error"] = "Emergency already assigned"
            return False, info
        
        # Check hospital exists and has capacity
        if not (1 <= hospital_id <= self.num_hospitals):
            info["hospital_error"] = "Invalid hospital ID"
            return False, info
        
        hospital = self.hospitals[hospital_id - 1]
        if hospital["current_capacity"] <= 0:
            info["hospital_error"] = "Hospital at capacity"
            return False, info
        
        return True, info
    
    def _execute_assignment(self, ambulance_id: int, emergency_id: int, hospital_id: int) -> float:
        """
        Execute assignment and calculate reward.
        
        Reward Function Components:
        - priority_handling (0.5): High-severity cases get priority
        - response_speed (0.3): Quick response reduces penalties
        - resource_usage (0.2): Efficient ambulance/hospital use
        
        Returns:
            Total reward for this step
        """
        ambulance = self.ambulances[ambulance_id - 1]
        emergency = self.emergencies[emergency_id - 1]
        hospital = self.hospitals[hospital_id - 1]
        
        # Calculate travel distance and response time
        ambulance_distance = abs(ambulance["location"] - emergency["location"])
        travel_time = ambulance_distance * self.traffic_factor / 2  # Normalize by 2
        response_time = emergency["time_waiting"] + travel_time
        
        reward = 0.0
        
        # 1. Priority Handling Reward (0.5 weight)
        # High-severity emergencies should be handled first
        severity_score = emergency["severity"] / 10.0  # Normalize to 0-1
        priority_reward = severity_score * 0.5
        
        # Penalty for handling low-severity when high-severity waiting
        for e in self.emergencies:
            if e["severity"] > emergency["severity"] and e["assigned_ambulance"] is None:
                priority_reward -= 0.5  # Significant penalty
                break
        
        reward += priority_reward
        
        # 2. Response Speed Reward (0.3 weight)
        # Reward for fast response, penalize delays
        speed_penalty = min(emergency["time_waiting"] * 0.05, 0.3)
        speed_reward = 0.3 - speed_penalty
        reward += speed_reward
        
        # 3. Resource Usage Reward (0.2 weight)
        # Reward for using resources efficiently
        hospital_utilization = hospital["patients"] / hospital["max_capacity"]
        resource_reward = (1 - hospital_utilization) * 0.2
        reward += resource_reward
        
        # Mark assignment
        emergency["assigned_ambulance"] = ambulance_id
        emergency["assigned_hospital"] = hospital_id
        ambulance["available"] = False
        ambulance["busy_until"] = travel_time + 5  # Recovery time
        ambulance["current_emergency"] = emergency_id
        hospital["current_capacity"] -= 1
        hospital["patients"] += 1
        
        # Track metrics
        self.total_response_time += response_time
        if emergency["severity"] >= 8:
            self.high_severity_handled += 1
        
        return min(max(reward, -1.0), 1.0)  # Clamp reward to [-1, 1]
    
    def _update_environment(self):
        """Update environment state each step."""
        # Increase waiting time for unassigned emergencies
        for emergency in self.emergencies:
            if emergency["assigned_ambulance"] is None:
                emergency["time_waiting"] += 1
                self.total_high_severity += 1 if emergency["severity"] >= 8 else 0
        
        # Decrease ambulance busy time
        for ambulance in self.ambulances:
            if ambulance["busy_until"] > 0:
                ambulance["busy_until"] -= 1
                if ambulance["busy_until"] <= 0:
                    ambulance["available"] = True
                    ambulance["current_emergency"] = None
        
        # Random traffic level change
        if random.random() < 0.1:
            self.traffic_level = max(1, min(5, self.traffic_level + random.randint(-1, 1)))
        
        # Random new emergency spawn
        if random.random() < self.emergency_spawn_rate:
            new_emergency = {
                "id": len(self.emergencies) + 1,
                "severity": random.randint(1, 10),
                "location": random.randint(0, self.grid_size - 1),
                "time_waiting": 0,
                "assigned_ambulance": None,
                "assigned_hospital": None
            }
            self.emergencies.append(new_emergency)
    
    def _all_emergencies_handled(self) -> bool:
        """Check if all emergencies are handled."""
        return all(e["assigned_ambulance"] is not None for e in self.emergencies)
    
    def state(self) -> Dict[str, Any]:
        """
        Get current environment state (OpenEnv compatible).
        
        Returns:
            Dictionary containing:
            - emergencies: List of emergency dicts
            - ambulances: List of ambulance dicts
            - hospitals: List of hospital dicts
            - traffic_level: Current traffic level
        """
        return self._get_state()
    
    def render(self, verbose: bool = False):
        """
        Render environment state for visualization and debugging.
        
        Args:
            verbose: If True, print all emergencies and detailed metrics
        """
        state = self._get_state()
        traffic_bar = "#" * state['traffic_level'] + "-" * (5 - state['traffic_level'])
        print(f"\n{'='*70}")
        print(f"Step {self.step_count:3d} | Task: {self.task_difficulty.upper()} | Traffic: {traffic_bar}")
        print(f"{'='*70}")
        
        # Emergencies status
        print(f"\n[EMERGENCIES] ({len(state['emergencies'])} total)")
        emergencies_to_show = state['emergencies'] if verbose else state['emergencies'][:4]
        for e in emergencies_to_show:
            status_icon = "DONE" if e['assigned'] else "WAIT"
            severity_bar = "[" + "#" * e['severity'] + "-" * (10 - e['severity']) + "]"
            print(f"  {status_icon} E{e['id']}: {severity_bar} Severity {e['severity']}/10 | Waiting {e['time_waiting']}s")
        
        if not verbose and len(state['emergencies']) > 4:
            print(f"  ... and {len(state['emergencies']) - 4} more emergencies")
        
        # Ambulances status
        available_count = sum(1 for a in state['ambulances'] if a['available'])
        print(f"\n[AMBULANCES] ({available_count}/{len(state['ambulances'])} available)")
        for a in state['ambulances']:
            status = "Ready" if a['available'] else f"Busy ({a['busy_until']}s)"
            location_marker = f"@L{a['location']}"
            print(f"  A{a['id']}: {status:15} {location_marker}")
        
        # Hospitals status
        print(f"\n[HOSPITALS] (Capacity Status)")
        for h in state['hospitals']:
            utilization = h['patients']
            total = state.get('max_capacity', 5)
            bar = "[" + "#" * utilization + "-" * (total - utilization) + "]"
            print(f"  H{h['id']}: {bar} {h['patients']} patients | {h['capacity']} beds available | @L{h['location']}")
        
        # Performance metrics
        print(f"\n[METRICS]")
        print(f"  Response Time (Avg): {self.total_response_time / max(1, self.high_severity_handled):.1f}s")
        print(f"  High-Severity Handled: {self.high_severity_handled}/{self.total_high_severity}")
        assigned_count = sum(1 for e in state['emergencies'] if e['assigned'])
        print(f"  Assignments: {assigned_count}/{len(state['emergencies'])}")
        print(f"{'-'*70}\n")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive episode metrics.
        
        Returns:
            Dictionary with all tracked metrics
        """
        state = self._get_state()
        assigned_emergencies = [e for e in self.emergencies if e['assigned_ambulance'] is not None]
        unassigned_emergencies = [e for e in self.emergencies if e['assigned_ambulance'] is None]
        
        return {
            "step_count": self.step_count,
            "total_response_time": self.total_response_time,
            "avg_response_time": self.total_response_time / max(1, len(assigned_emergencies)),
            "high_severity_handled": self.high_severity_handled,
            "total_high_severity": self.total_high_severity,
            "emergencies_assigned": len(assigned_emergencies),
            "emergencies_unassigned": len(unassigned_emergencies),
            "ambulances_available": sum(1 for a in self.ambulances if a['available']),
            "ambulances_busy": sum(1 for a in self.ambulances if not a['available']),
            "hospital_total_patients": sum(h['patients'] for h in self.hospitals),
            "hospital_total_capacity": sum(h['current_capacity'] for h in self.hospitals),
            "avg_wait_time": np.mean([e['time_waiting'] for e in unassigned_emergencies]) if unassigned_emergencies else 0.0,
            "traffic_level": self.traffic_level
        }
    
    def set_task_parameters(self, **kwargs):
        """
        Set custom task parameters for advanced usage.
        
        Supported parameters:
            - num_emergencies: Number of emergencies
            - hospital_capacity: Capacity per hospital
            - traffic_factor: Traffic delay multiplier
            - initial_busy_ambulances: Number of initially busy ambulances
            - emergency_spawn_rate: Rate of new emergency spawning
        
        Example:
            env.set_task_parameters(
                num_emergencies=10,
                hospital_capacity=2,
                traffic_factor=2.5
            )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current episode.
        
        Returns:
            Dictionary with episode statistics and performance analysis
        """
        state = self._get_state()
        assigned_emergencies = [e for e in self.emergencies if e['assigned_ambulance'] is not None]
        unassigned_emergencies = [e for e in self.emergencies if e['assigned_ambulance'] is None]
        
        # Calculate statistics
        avg_response_time = (self.total_response_time / len(assigned_emergencies)) if assigned_emergencies else 0.0
        high_severity_percentage = (self.high_severity_handled / self.total_high_severity) if self.total_high_severity > 0 else 0.0
        
        # Calculate resource utilization
        ambulance_utilization = len([a for a in self.ambulances if not a['available']]) / len(self.ambulances)
        hospital_utilization = sum(h['patients'] for h in self.hospitals) / sum(h['max_capacity'] for h in self.hospitals) if self.hospitals else 0.0
        
        return {
            "task_difficulty": self.task_difficulty,
            "total_steps": self.step_count,
            "total_response_time": self.total_response_time,
            "avg_response_time": avg_response_time,
            "high_severity_handled": self.high_severity_handled,
            "total_high_severity": self.total_high_severity,
            "high_severity_percentage": high_severity_percentage,
            "emergencies_assigned": len(assigned_emergencies),
            "emergencies_unassigned": len(unassigned_emergencies),
            "total_emergencies": len(self.emergencies),
            "ambulance_utilization": ambulance_utilization,
            "hospital_utilization": hospital_utilization,
            "avg_emergency_wait_time": np.mean([e['time_waiting'] for e in unassigned_emergencies]) if unassigned_emergencies else 0.0,
            "max_emergency_wait_time": max([e['time_waiting'] for e in unassigned_emergencies], default=0),
            "traffic_level_final": self.traffic_level
        }
    
    def record_episode(self, agent_name: str = "unknown"):
        """
        Record episode to history for trend analysis.
        
        Args:
            agent_name: Name of agent that ran this episode
        """
        summary = self.get_episode_summary()
        summary["agent_name"] = agent_name
        self.episode_history.append(summary)
    
    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Get history of all recorded episodes."""
        return self.episode_history
    
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends across episode history.
        
        Returns:
            Trend analysis including averages and improvements
        """
        if not self.episode_history:
            return {"message": "No episode history available"}
        
        scores = [e.get('high_severity_percentage', 0) for e in self.episode_history]
        response_times = [e.get('avg_response_time', 0) for e in self.episode_history]
        resource_utils = [e.get('hospital_utilization', 0) for e in self.episode_history]
        
        return {
            "num_episodes": len(self.episode_history),
            "avg_high_severity_score": float(np.mean(scores)) if scores else 0.0,
            "avg_response_time": float(np.mean(response_times)) if response_times else 0.0,
            "avg_resource_utilization": float(np.mean(resource_utils)) if resource_utils else 0.0,
            "improvement_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "best_episode": max(self.episode_history, key=lambda x: x.get('high_severity_percentage', 0)) if self.episode_history else None,
            "worst_episode": min(self.episode_history, key=lambda x: x.get('high_severity_percentage', 0)) if self.episode_history else None
        }
    
    def validate_action_detailed(self, ambulance_id: int, emergency_id: int, hospital_id: int) -> Tuple[bool, List[str]]:
        """
        Validate action with detailed error messages.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check ambulance
        if not (1 <= ambulance_id <= self.num_ambulances):
            errors.append(f"Ambulance ID {ambulance_id} out of range [1-{self.num_ambulances}]")
        elif not self.ambulances[ambulance_id - 1]['available']:
            errors.append(f"Ambulance {ambulance_id} is not available")
        
        # Check emergency
        if not (1 <= emergency_id <= len(self.emergencies)):
            errors.append(f"Emergency ID {emergency_id} out of range [1-{len(self.emergencies)}]")
        elif self.emergencies[emergency_id - 1]['assigned_ambulance'] is not None:
            errors.append(f"Emergency {emergency_id} is already assigned")
        
        # Check hospital
        if not (1 <= hospital_id <= self.num_hospitals):
            errors.append(f"Hospital ID {hospital_id} out of range [1-{self.num_hospitals}]")
        elif self.hospitals[hospital_id - 1]['current_capacity'] <= 0:
            errors.append(f"Hospital {hospital_id} is at full capacity")
        
        return len(errors) == 0, errors
