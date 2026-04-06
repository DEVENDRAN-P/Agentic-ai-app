#!/usr/bin/env python3
"""
OpenEnv Specification Validator
Ensures project meets all OpenEnv requirements
"""

import sys
import json
from pathlib import Path

def validate():
    errors = []
    warnings = []
    
    print("🔍 OpenEnv Specification Validation\n")
    
    # 1. Check core files
    print("1️⃣  Checking core files...")
    required_files = [
        "src/env.py",
        "src/inference.py",
        "src/graders.py",
        "README.md",
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            errors.append(f"Missing: {file}")
            print(f"   ❌ {file}")
    
    # Check for openenv.yaml in root or configs/
    openenv_yaml_path = None
    if Path("openenv.yaml").exists():
        openenv_yaml_path = "openenv.yaml"
        print(f"   ✅ openenv.yaml (at root)")
    elif Path("configs/openenv.yaml").exists():
        openenv_yaml_path = "configs/openenv.yaml"
        print(f"   ✅ openenv.yaml (at configs/)")
    else:
        errors.append("Missing: openenv.yaml")
        print(f"   ❌ openenv.yaml (not found at root or configs/)")
    
    # 2. Check environment implementation
    print("\n2️⃣  Checking environment implementation...")
    env = None
    try:
        from src.env import EmergencyResponseEnv, Observation, Action, Reward
        print("   ✅ Environment class found")
        print("   ✅ Observation, Action, Reward models found")
        
        # Test instantiation
        env = EmergencyResponseEnv()
        print("   ✅ Environment instantiates")
        
        # Test reset
        obs = env.reset()
        print(f"   ✅ reset() works (returns {type(obs).__name__})")
        
        # Test state
        state = env.state()
        print(f"   ✅ state() works (returns dict with keys: {list(state.keys())[:3]}...)")
        
        # Test step with dict (not Action object)
        action = {"ambulance_id": 1, "emergency_id": 1, "hospital_id": 1}
        obs, reward, done, info = env.step(action)
        print(f"   ✅ step() works (returns tuple of 4 elements)")
        print(f"   ✅ Reward structure valid (value={reward}, type={type(reward).__name__})")
        
    except Exception as e:
        errors.append(f"Environment error: {str(e)}")
        print(f"   ❌ {e}")
    
    # 3. Check OpenEnv config
    print("\n3️⃣  Checking OpenEnv configuration...")
    try:
        if not openenv_yaml_path:
            raise FileNotFoundError("openenv.yaml not found")
        
        import yaml
        with open(openenv_yaml_path) as f:
            config = yaml.safe_load(f)
        
        if config.get("version") == "1.0":
            print("   ✅ Version 1.0 specified")
        else:
            warnings.append(f"Unexpected version: {config.get('version')}")
        
        tasks = config.get("environment", {}).get("tasks", {})
        required_tasks = ["easy", "medium", "hard"]
        for task in required_tasks:
            if task in tasks:
                print(f"   ✅ Task '{task}' defined")
            else:
                errors.append(f"Missing task: {task}")
                print(f"   ❌ Task '{task}' missing")
        
        # Check schemas
        if "observation_schema" in config:
            print("   ✅ observation_schema defined")
        if "action_schema" in config:
            print("   ✅ action_schema defined")
        if "reward_schema" in config:
            print("   ✅ reward_schema defined")
            
    except Exception as e:
        errors.append(f"Config validation error: {str(e)}")
        print(f"   ❌ {e}")
    
    # 4. Check grader
    print("\n4️⃣  Checking grader implementation...")
    try:
        from src.graders import EmergencyResponseGrader
        grader = EmergencyResponseGrader()
        print("   ✅ EmergencyResponseGrader class found and instantiates")
        
        # Test evaluate_episode
        if hasattr(grader, 'evaluate_episode'):
            print("   ✅ evaluate_episode method exists")
        else:
            warnings.append("Grader missing evaluate_episode")
            
    except Exception as e:
        errors.append(f"Grader error: {str(e)}")
        print(f"   ❌ {e}")
    
    # 5. Check inference
    print("\n5️⃣  Checking inference implementation...")
    try:
        from src.inference import SmartHeuristicAgent, QLearningAgent, RandomBaselineAgent
        print("   ✅ SmartHeuristicAgent found")
        print("   ✅ QLearningAgent found")
        print("   ✅ RandomBaselineAgent found")
        
        # Test agent instantiation (need env)
        if env:
            agent = SmartHeuristicAgent(env=env)
            print("   ✅ SmartHeuristicAgent instantiates with environment")
        else:
            warnings.append("Could not test agent instantiation without env")
        
    except Exception as e:
        errors.append(f"Inference error: {str(e)}")
        print(f"   ❌ {e}")
    
    # 6. Check CLI interface
    print("\n6️⃣  Checking CLI interface...")
    try:
        if Path("inference.py").exists():
            print("   ✅ inference.py CLI found")
            
            # Try importing it
            from inference import main
            print("   ✅ main() function found")
        else:
            warnings.append("CLI interface not found")
            
    except Exception as e:
        warnings.append(f"CLI check: {str(e)}")
    
    # 7. Check documentation
    print("\n7️⃣  Checking documentation...")
    try:
        with open("README.md", encoding="utf-8", errors="ignore") as f:
            readme = f.read()
        
        required_sections = [
            "Emergency Response",
            "observation",
            "action",
            "reward",
        ]
        
        for section in required_sections:
            if section.lower() in readme.lower():
                print(f"   ✅ Documentation includes '{section}'")
            else:
                warnings.append(f"README missing section: {section}")
                
    except Exception as e:
        errors.append(f"Documentation check: {str(e)}")
        print(f"   ❌ {e}")
    
    # Summary
    print("\n" + "="*50)
    if not errors:
        print("✅ OPENENV VALIDATION PASSED")
        print("="*50)
        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for w in warnings:
                print(f"   - {w}")
        return 0
    else:
        print(f"❌ OPENENV VALIDATION FAILED")
        print("="*50)
        print(f"\n❌ {len(errors)} error(s):")
        for e in errors:
            print(f"   - {e}")
        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for w in warnings:
                print(f"   - {w}")
        return 1

if __name__ == "__main__":
    sys.exit(validate())
