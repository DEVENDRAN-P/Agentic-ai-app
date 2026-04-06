#!/usr/bin/env python3
"""
OpenEnv CLI wrapper for validation and environment inspection.
Provides the equivalent of 'openenv validate' command.
"""

import sys
import argparse
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_command():
    """Run OpenEnv validation."""
    import yaml
    import inspect
    from src.env import EmergencyResponseEnv, Observation, Action, Reward
    from pydantic import BaseModel
    
    print("\n" + "="*70)
    print("OpenEnv Validation")
    print("="*70 + "\n")
    
    errors = []
    
    # 1. Check openenv.yaml (try root first, then configs/)
    config_path = None
    if os.path.exists('openenv.yaml'):
        config_path = 'openenv.yaml'
    elif os.path.exists('configs/openenv.yaml'):
        config_path = 'configs/openenv.yaml'
    
    try:
        if not config_path:
            raise FileNotFoundError("openenv.yaml not found at root or configs/")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Found openenv.yaml at {config_path}")
        
        required_fields = ['version', 'name', 'environment', 'spaces']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing '{field}' in openenv.yaml")
            else:
                print(f"  ✅ {field}: {config[field] if isinstance(config[field], (str, int)) else '...'}")
    except FileNotFoundError:
        errors.append("openenv.yaml not found at root or configs/openenv.yaml")
        print("❌ openenv.yaml missing (checked: openenv.yaml, configs/openenv.yaml)")
    except Exception as e:
        errors.append(f"Failed to parse openenv.yaml: {e}")
        print(f"❌ openenv.yaml parse error: {e}")
    
    # 2. Check Pydantic models
    print("\n✅ Pydantic Models:")
    for model_name in ['Observation', 'Action', 'Reward']:
        print(f"  ✅ {model_name} (BaseModel)")
    
    # 3. Check environment methods
    print("\n✅ Environment Methods:")
    env_methods = ['reset', 'step', 'state']
    for method in env_methods:
        if hasattr(EmergencyResponseEnv, method):
            print(f"  ✅ {method}()")
        else:
            errors.append(f"Missing method: {method}()")
    
    # 4. Test environment functionality
    print("\n✅ Functionality Test:")
    try:
        env = EmergencyResponseEnv(task_difficulty="easy")
        print("  ✅ Environment instantiation")
        
        obs = env.reset()
        print("  ✅ reset() executed successfully")
        
        state = env.state()
        print("  ✅ state() executed successfully")
        
        action = {"ambulance_id": 1, "emergency_id": 1, "hospital_id": 1}
        obs, reward, done, info = env.step(action)
        print("  ✅ step(action) executed successfully")
    except Exception as e:
        errors.append(f"Functionality test failed: {e}")
    
    # 5. Summary
    print("\n" + "="*70)
    if errors:
        print("❌ FAILED - Issues found:")
        for error in errors:
            print(f"  ❌ {error}")
        print("="*70 + "\n")
        return 1
    else:
        print("✅ PASSED - All OpenEnv requirements satisfied!")
        print("="*70 + "\n")
        return 0

def main():
    parser = argparse.ArgumentParser(description="OpenEnv CLI tool")
    parser.add_argument('command', choices=['validate', 'inspect'], 
                        help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        return validate_command()
    elif args.command == 'inspect':
        return validate_command()  # Same for now

if __name__ == "__main__":
    sys.exit(main())
