from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import os
import sys
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Emergency Response AI Environment",
    description="OpenEnv-compliant emergency response optimization environment",
    version="2.0"
)

# Requirement 7: Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Global environment state for reset/step flow
_current_env: Optional[Any] = None
_current_difficulty: str = "easy"


@app.get("/")
def root() -> dict:
    """Root endpoint - Requirement 2: Respond with 200 status"""
    return {
        "status": "running",
        "message": "Emergency Response AI Environment - Hackathon Submission",
        "version": "1.0",
        "endpoints": {
            "reset": {
                "method": "GET",
                "path": "/reset/{difficulty}",
                "params": ["easy", "medium", "hard"],
                "returns": "observation"
            },
            "step": {
                "method": "POST",
                "path": "/step",
                "body": {"ambulance_id": "int", "emergency_id": "int", "hospital_id": "int"},
                "returns": "observation, reward, done"
            },
            "state": {
                "method": "GET",
                "path": "/state",
                "returns": "current observation"
            },
            "ping": {
                "method": "GET",
                "path": "/ping",
                "returns": "health check"
            },
            "validate": {
                "method": "GET",
                "path": "/validate",
                "returns": "OpenEnv compliance check"
            },
            "run": {
                "method": "GET",
                "path": "/run",
                "params": ["task", "episodes"],
                "returns": "inference results"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "returns": "service health"
            }
        }
    }


@app.get("/ping")
def ping() -> dict:
    """Automated ping endpoint - Requirement 2: Must respond with 200 and reset()"""
    try:
        from src.env import EmergencyResponseEnv
        
        # Create env and call reset() - shows reset() works
        env = EmergencyResponseEnv(task_difficulty="easy")
        state = env.reset()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Environment reset successful",
                "state_keys": list(state.keys()),
                "timestamp": str(__import__('datetime').datetime.now())
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )


@app.api_route("/reset", methods=["GET", "POST"])
def reset_default() -> dict:
    """Reset endpoint - Default to easy (GET or POST)"""
    return reset_with_difficulty("easy")


@app.api_route("/reset/{difficulty}", methods=["GET", "POST"])
def reset_with_difficulty(difficulty: str = "easy") -> dict:
    """Reset endpoint - Requirement 2: Callable reset() function - Returns observation
    
    Supports both GET and POST methods.
    Query Parameters or Body:
        difficulty: str - "easy", "medium", or "hard"
    
    Returns:
        {
            "observation": {...},  # Current state
            "status": "success"
        }
    """
    global _current_env, _current_difficulty
    
    try:
        valid_difficulties = ["easy", "medium", "hard"]
        if difficulty not in valid_difficulties:
            difficulty = "easy"
        
        from src.env import EmergencyResponseEnv
        
        # Create new environment and reset it
        _current_env = EmergencyResponseEnv(task_difficulty=difficulty)
        _current_difficulty = difficulty
        observation = _current_env.reset()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Environment reset for {difficulty} task",
                "observation": {
                    "num_emergencies": len(observation.get("emergencies", [])),
                    "num_ambulances": len(observation.get("ambulances", [])),
                    "num_hospitals": len(observation.get("hospitals", [])),
                    "traffic_level": observation.get("traffic_level", 0),
                    "step": observation.get("step", 0)
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )


@app.post("/step")
async def step(action: Dict[str, Any]) -> dict:
    """Step endpoint - Requirement 5: Take action and get (observation, reward, done)
    
    Request Body (POST):
        {
            "ambulance_id": int,
            "emergency_id": int,
            "hospital_id": int
        }
    
    Returns:
        {
            "observation": {...},   # New state after action
            "reward": float,        # Reward for this action
            "done": bool,           # Episode finished?
            "info": {...}           # Additional info
        }
    """
    global _current_env
    
    try:
        if _current_env is None:
            raise ValueError("Environment not initialized. Call /reset first.")
        
        # Execute step
        next_state, reward, done, info = _current_env.step(action)
        
        return {
            "status": "success",
            "observation": {
                "num_emergencies": len(next_state.get("emergencies", [])),
                "num_ambulances": len(next_state.get("ambulances", [])),
                "num_hospitals": len(next_state.get("hospitals", [])),
                "traffic_level": next_state.get("traffic_level", 0),
                "step": next_state.get("step", 0)
            },
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state() -> dict:
    """State endpoint - Returns current observation without taking action
    
    Returns:
        {
            "observation": {...},   # Current state
            "status": "success"
        }
    """
    global _current_env
    
    try:
        if _current_env is None:
            raise ValueError("Environment not initialized. Call /reset first.")
        
        current_state = _current_env.state()
        
        return {
            "status": "success",
            "observation": {
                "num_emergencies": len(current_state.get("emergencies", [])),
                "num_ambulances": len(current_state.get("ambulances", [])),
                "num_hospitals": len(current_state.get("hospitals", [])),
                "traffic_level": current_state.get("traffic_level", 0),
                "step": current_state.get("step", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate")
def validate() -> dict:
    """OpenEnv validation endpoint - Requirement 3: openenv validate compliance"""
    try:
        from src.env import EmergencyResponseEnv
        from src.graders import create_grader_for_task, EasyTaskGrader, MediumTaskGrader, HardTaskGrader
        
        validation_results = {
            "openenv_compliance": "CHECKING",
            "steps": []
        }
        
        # Check 1: Environment has reset(), step(), state()
        env = EmergencyResponseEnv(task_difficulty="easy")
        assert hasattr(env, 'reset'), "Missing reset() method"
        assert hasattr(env, 'step'), "Missing step() method"
        assert hasattr(env, 'state'), "Missing state() method"
        validation_results["steps"].append("✅ reset(), step(), state() methods present")
        
        # Check 2: reset() returns dict
        state = env.reset()
        assert isinstance(state, dict), "reset() must return dict"
        validation_results["steps"].append("✅ reset() returns Dict")
        
        # Check 3: step() works properly
        action = {
            "ambulance_id": 1,
            "emergency_id": 1,
            "hospital_id": 1
        }
        next_state, reward, done, info = env.step(action)
        assert isinstance(reward, (int, float)), "Reward must be numeric"
        assert 0.0 <= reward <= 1.0 or reward == -0.40, "Reward must be in [0,1] or -0.40"
        validation_results["steps"].append("✅ step() works and returns valid reward")
        
        # Check 4: Graders exist and work
        for difficulty in ["easy", "medium", "hard"]:
            grader = create_grader_for_task(difficulty)
            assert grader is not None, f"No grader for {difficulty}"
            validation_results["steps"].append(f"✅ {difficulty.capitalize()} grader exists")
        
        # Check 5: Graders return scores in [0, 1]
        grader = create_grader_for_task("easy")
        env = EmergencyResponseEnv(task_difficulty="easy")
        state = env.reset()
        score = grader.evaluate_episode(env, [])
        assert 0.0 <= score.get("final_score", 0) <= 1.0, "Score must be in [0, 1]"
        validation_results["steps"].append("✅ Scores in range [0.0, 1.0]")
        
        validation_results["openenv_compliance"] = "PASSED"
        validation_results["status"] = "all_checks_passed"
        
        return JSONResponse(
            status_code=200,
            content=validation_results
        )
    except AssertionError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "validation_failed", "error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.get("/run")
def run(task: str = "easy", episodes: int = 1) -> dict:
    """Run inference - Requirement 5: Baseline must reproduce"""
    try:
        valid_tasks = ["easy", "medium", "hard"]
        if task not in valid_tasks:
            task = "easy"
        
        # Cap episodes to avoid timeout
        episodes = min(episodes, 5)
        
        result = subprocess.run(
            ["python", "inference.py", "--task", task, "--episodes", str(episodes), "--agent", "heuristic"],
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout per run (requirement 10: 20 min total)
        )
        
        return JSONResponse(
            status_code=200 if result.returncode == 0 else 500,
            content={
                "status": "success" if result.returncode == 0 else "failed",
                "task": task,
                "episodes": episodes,
                "output": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
                "error": result.stderr[-500:] if result.stderr else "",
                "return_code": result.returncode
            }
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={"status": "timeout", "error": "Inference exceeded 5-minute timeout"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.get("/health")
def health() -> dict:
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "service": "emergency-response-ai",
        "version": "1.0"
    }


def main():
    """Main entry point for multi-mode deployment"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
