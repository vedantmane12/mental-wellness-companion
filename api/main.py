"""
FastAPI backend for Mental Wellness Companion
Provides REST API for the RL-powered mental health support system
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
import uuid

from src.agents.orchestrator import Orchestrator
from src.simulation.persona_generator import PersonaGenerator
from src.simulation.user_simulator import UserSimulator
from src.safety.safety_monitor import SafetyMonitor
from src.utils.logger import logger
from config.settings import DATA_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Mental Wellness Companion API",
    description="RL-powered mental health support system using PPO and Contextual Bandits",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
orchestrator = Orchestrator()
persona_generator = PersonaGenerator()
user_simulator = UserSimulator()
safety_monitor = SafetyMonitor()

# Store active sessions
active_sessions = {}

# Pydantic models
class SessionCreate(BaseModel):
    persona_id: Optional[str] = None
    use_random_persona: bool = True

class MessageInput(BaseModel):
    session_id: str
    message: str

class TrainingRequest(BaseModel):
    episodes: int = 100
    batch_size: int = 32

class EvaluationRequest(BaseModel):
    num_personas: int = 10

class SessionResponse(BaseModel):
    session_id: str
    persona: Dict[str, Any]
    initial_message: str
    status: str

class MessageResponse(BaseModel):
    response: str
    strategy: str
    resource: Dict[str, Any]
    metrics: Dict[str, float]
    safety_check: Dict[str, Any]
    continue_conversation: bool

class TrainingStatus(BaseModel):
    status: str
    episodes_completed: int
    current_reward: float
    best_reward: float
    training_metrics: Dict[str, Any]

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Mental Wellness Companion API",
        "version": "1.0.0",
        "description": "RL-powered mental health support using PPO and Contextual Bandits",
        "endpoints": {
            "POST /session/create": "Create new conversation session",
            "POST /chat": "Send message in conversation",
            "GET /session/{session_id}": "Get session details",
            "GET /training/status": "Get training status",
            "POST /training/start": "Start training",
            "GET /evaluation/results": "Get evaluation results",
            "GET /statistics": "Get system statistics"
        }
    }

@app.post("/session/create", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new conversation session"""
    try:
        # Generate or load persona
        if request.use_random_persona:
            personas = persona_generator.generate_batch(1, save=False)
            persona = personas[0]
        else:
            # Load existing persona
            existing_personas = persona_generator.load_personas("training")
            if not existing_personas:
                raise HTTPException(status_code=404, detail="No personas available")
            persona = existing_personas[0]
        
        # Create session
        session_id = str(uuid.uuid4())
        user_session_id = user_simulator.create_session(persona)
        
        # Get initial message
        initial_message = user_simulator.get_initial_message(user_session_id)
        
        # Store session info
        active_sessions[session_id] = {
            "user_session_id": user_session_id,
            "persona": persona,
            "created_at": datetime.now().isoformat(),
            "turn_count": 0,
            "conversation_history": []
        }
        
        logger.info(f"Created session {session_id} for persona {persona['id']}")
        
        return SessionResponse(
            session_id=session_id,
            persona=persona,
            initial_message=initial_message,
            status="active"
        )
    
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageInput):
    """Process a chat message"""
    try:
        # Validate session
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[request.session_id]
        
        # Safety check on user message
        is_safe, violation = safety_monitor.check_message(request.message)
        safety_info = {
            "is_safe": is_safe,
            "violation": str(violation) if violation else None
        }
        
        context = {
            "conversation_history": session["conversation_history"],
            "training": False,
            "exploration": 0.3,  # Add exploration parameter
            "force_different_strategy": len(session["conversation_history"]) > 2  # Force variety
        }
        
        # Get orchestrated response
        response = orchestrator.process_interaction(
            session["user_session_id"],
            request.message,
            session["persona"], 
            context
        )
        
        # Safety check on agent response
        is_appropriate, modified_response = safety_monitor.check_response(
            response["message"],
            response.get("strategy", "supportive"),
            {"risk_level": response["monitoring"]["risk_level"]}
        )
        
        # Update session
        session["turn_count"] += 1
        session["conversation_history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        session["conversation_history"].append({
            "role": "assistant",
            "content": modified_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "strategy": response.get("strategy"),
                "resource": response.get("resource")
            }
        })
        
        # Prepare response
        return MessageResponse(
            response=modified_response,
            strategy=response.get("strategy", "supportive"),
            resource=response.get("resource", {}),
            metrics={
                "risk_level": response["monitoring"]["risk_level"],
                "health_score": response["monitoring"]["health_score"],
                "turn_count": session["turn_count"]
            },
            safety_check=safety_info,
            continue_conversation=response.get("continue_conversation", True)
        )
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Get session summary from user simulator
    summary = user_simulator.get_session_summary(session["user_session_id"])
    
    return {
        "session_id": session_id,
        "persona": session["persona"],
        "turn_count": session["turn_count"],
        "created_at": session["created_at"],
        "conversation_history": session["conversation_history"],
        "summary": summary
    }

@app.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    try:
        # Load training stats
        stats_path = DATA_DIR / "training_stats.json"
        if not stats_path.exists():
            return TrainingStatus(
                status="not_started",
                episodes_completed=0,
                current_reward=0.0,
                best_reward=0.0,
                training_metrics={}
            )
        
        with open(stats_path) as f:
            stats = json.load(f)
        
        return TrainingStatus(
            status="completed",
            episodes_completed=len(stats.get("episode_rewards", [])),
            current_reward=stats.get("episode_rewards", [0])[-1] if stats.get("episode_rewards") else 0,
            best_reward=stats.get("best_reward", 0),
            training_metrics={
                "avg_reward": sum(stats.get("episode_rewards", [])[-10:]) / min(10, len(stats.get("episode_rewards", [1]))) if stats.get("episode_rewards") else 0,
                "policy_loss": stats.get("training_stats", {}).get("policy_losses", [0])[-1] if stats.get("training_stats", {}).get("policy_losses") else 0,
                "value_loss": stats.get("training_stats", {}).get("value_losses", [0])[-1] if stats.get("training_stats", {}).get("value_losses") else 0
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start training process (simplified for demo)"""
    return {
        "status": "Training initiated",
        "message": "Please run 'python scripts/train.py' to start full training",
        "config": {
            "episodes": request.episodes,
            "batch_size": request.batch_size
        }
    }

@app.get("/evaluation/results")
async def get_evaluation_results():
    """Get evaluation results"""
    try:
        results_path = DATA_DIR / "evaluation_results.json"
        if not results_path.exists():
            return {"status": "No evaluation results available"}
        
        with open(results_path) as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive system statistics"""
    try:
        stats = {
            "orchestrator": orchestrator.get_statistics(),
            "safety_monitor": safety_monitor.get_statistics(),
            "active_sessions": len(active_sessions),
            "total_sessions": orchestrator.stats["total_interactions"]
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personas")
async def get_personas(limit: int = 10):
    """Get available personas"""
    try:
        personas = persona_generator.load_personas("training")[:limit]
        
        return {
            "count": len(personas),
            "personas": [
                {
                    "id": p["id"],
                    "age": p["demographics"]["age"],
                    "occupation": p["demographics"]["occupation"],
                    "primary_concerns": p["mental_health"]["primary_concerns"]
                }
                for p in personas
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting personas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time conversation"""
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            response = await chat(MessageInput(
                session_id=session_id,
                message=message_data["message"]
            ))
            
            # Send response
            await websocket.send_json({
                "type": "agent_response",
                "data": response.dict()
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a conversation session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Get session summary
    summary = orchestrator.end_session(session["user_session_id"])
    
    # Clean up
    user_simulator.reset_session(session["user_session_id"])
    del active_sessions[session_id]
    
    return {
        "status": "session_ended",
        "summary": summary
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run with: uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)