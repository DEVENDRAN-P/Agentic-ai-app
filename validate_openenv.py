#!/usr/bin/env python3
"""
OpenEnv Validation Script
Checks if the project meets all OpenEnv hackathon requirements:
1. reset() - Starts a new episode and returns the first observation
2. step(action) - Takes an action and returns new observation, reward, done, info
3. state() - Returns current state of environment
4. Pydantic models for observation, action, reward
5. openenv.yaml file with proper configuration
"""

import sys
import os
from pathlib import Path
import yaml
import inspect

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_file_exists(filepath: str) -> tuple[bool, str]:
    """Check if a file exists."""
    if Path(filepath).exists():
        return True, f"✅ {filepath} exists"
    else:
        return False, f"❌ {filepath} is missing"

def check_openenv_yaml() -> tuple[bool, str]:
    """Check if openenv.yaml has required fields."""
    try:
        # Try root first, then configs/
        config_path = None
        if Path('openenv.yaml').exists():
            config_path = 'openenv.yaml'
        elif Path('configs/openenv.yaml').exists():
            config_path = 'configs/openenv.yaml'
        
        if not config_path:
            return False, f"❌ openenv.yaml not found (checked: openenv.yaml, configs/openenv.yaml)"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_fields = ['version', 'name', 'environment', 'spaces']
        for field in required_fields:
            if field not in config:
                return False, f"❌ openenv.yaml missing '{field}' field"
        
        return True, "✅ openenv.yaml has all required fields"
    except Exception as e:
        return False, f"❌ openenv.yaml validation failed: {e}"

def check_pydantic_models() -> tuple[bool, str]:
    """Check if Pydantic models exist."""
    try:
        from src.env import Observation, Action, Reward, Emergency, Ambulance, Hospital
        from pydantic import BaseModel
        
        models = [
            ('Observation', Observation),
            ('Action', Action),
            ('Reward', Reward),
            ('Emergency', Emergency),
            ('Ambulance', Ambulance),
            ('Hospital', Hospital)
        ]
        
        for name, model in models:
            if not issubclass(model, BaseModel):
                return False, f"❌ {name} is not a Pydantic BaseModel"
        
        return True, "✅ All Pydantic models defined (Observation, Action, Reward, Emergency, Ambulance, Hospital)"
    except Exception as e:
        return False, f"❌ Pydantic models check failed: {e}"

def check_env_methods() -> tuple[bool, list[str]]:
    """Check if EmergencyResponseEnv has reset(), step(), and state() methods."""
    try:
        from src.env import EmergencyResponseEnv
        
        errors = []
        
        # Check reset() method
        if not hasattr(EmergencyResponseEnv, 'reset'):
            errors.append("❌ EmergencyResponseEnv missing reset() method")
        else:
            reset_method = getattr(EmergencyResponseEnv, 'reset')
            sig = inspect.signature(reset_method)
            if len(sig.parameters) != 1:  # self only
                errors.append("❌ reset() has wrong signature")
        
        # Check step() method
        if not hasattr(EmergencyResponseEnv, 'step'):
            errors.append("❌ EmergencyResponseEnv missing step() method")
        else:
            step_method = getattr(EmergencyResponseEnv, 'step')
            sig = inspect.signature(step_method)
            if len(sig.parameters) != 2:  # self, action
                errors.append("❌ step() has wrong signature")
        
        # Check state() method
        if not hasattr(EmergencyResponseEnv, 'state'):
            errors.append("❌ EmergencyResponseEnv missing state() method")
        else:
            state_method = getattr(EmergencyResponseEnv, 'state')
            sig = inspect.signature(state_method)
            if len(sig.parameters) != 1:  # self only
                errors.append("❌ state() has wrong signature")
        
        if errors:
            return False, errors
        else:
            return True, ["✅ reset() method implemented", "✅ step(action) method implemented", "✅ state() method implemented"]
    
    except Exception as e:
        return False, [f"❌ Environment methods check failed: {e}"]

def test_env_workflow() -> tuple[bool, list[str]]:
    """Test that reset(), step(), and state() work correctly."""
    try:
        from src.env import EmergencyResponseEnv
        
        errors = []
        
        # Create environment
        env = EmergencyResponseEnv(task_difficulty="easy")
        
        # Test reset() returns observation
        obs = env.reset()
        if not isinstance(obs, dict):
            errors.append("❌ reset() did not return a dictionary")
        elif 'emergencies' not in obs or 'ambulances' not in obs or 'hospitals' not in obs:
            errors.append("❌ reset() observation missing required fields")
        else:
            errors.append("✅ reset() returns valid observation with emergencies, ambulances, hospitals")
        
        # Test state() returns observation
        state = env.state()
        if not isinstance(state, dict):
            errors.append("❌ state() did not return a dictionary")
        else:
            errors.append("✅ state() returns valid observation")
        
        # Test step() returns (observation, reward, done, info)
        action = {
            "ambulance_id": 1,
            "emergency_id": 1,
            "hospital_id": 1
        }
        result = env.step(action)
        
        if not isinstance(result, tuple) or len(result) != 4:
            errors.append(f"❌ step() returned {type(result)} instead of tuple(obs, reward, done, info)")
        else:
            obs, reward, done, info = result
            if not isinstance(obs, dict):
                errors.append("❌ step() observation is not a dict")
            elif not isinstance(reward, (int, float)):
                errors.append("❌ step() reward is not numeric")
            elif not isinstance(done, bool):
                errors.append("❌ step() done is not boolean")
            elif not isinstance(info, dict):
                errors.append("❌ step() info is not a dict")
            else:
                errors.append("✅ step(action) returns valid (observation, reward, done, info) tuple")
        
        if not errors or all('✅' in e for e in errors):
            return True, errors
        else:
            return False, errors
    
    except Exception as e:
        return False, [f"❌ Environment workflow test failed: {e}"]

def main():
    print("\n" + "="*70)
    print("OpenEnv Hackathon Validation")
    print("="*70 + "\n")
    
    checks = []
    all_passed = True
    
    # Check 1: openenv.yaml exists (try root, then configs/)
    config_exists = Path('openenv.yaml').exists() or Path('configs/openenv.yaml').exists()
    if config_exists:
        checks.append("✅ openenv.yaml found")
    else:
        checks.append("❌ openenv.yaml not found (checked: openenv.yaml, configs/openenv.yaml)")
        all_passed = False
    
    # Check 2: openenv.yaml structure
    if config_exists:
        valid, msg = check_openenv_yaml()
        checks.append(msg)
        all_passed = all_passed and valid
    
    # Check 3: Pydantic models
    valid, msg = check_pydantic_models()
    checks.append(msg)
    all_passed = all_passed and valid
    
    # Check 4: Environment methods exist
    valid, msgs = check_env_methods()
    checks.extend(msgs)
    all_passed = all_passed and valid
    
    # Check 5: Environment workflow
    valid, msgs = check_env_methods()
    checks.extend(msgs)
    all_passed = all_passed and valid
    
    # Check 6: Test environment workflow
    valid, msgs = test_env_workflow()
    checks.extend(msgs)
    all_passed = all_passed and valid
    
    # Print results
    print("\nValidation Checks:")
    print("-" * 70)
    for check in checks:
        print(check)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ PASS - All OpenEnv requirements satisfied!")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ FAIL - Some requirements not met")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
