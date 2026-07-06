from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
import logging
from fastapi.middleware.cors import CORSMiddleware

from app.models import GenerationRequest, GenerationResponse, HealthResponse
from app.ai_service import HuggingFaceService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for AI service
ai_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service
    
    logger.info("Starting up...")
    try:
        ai_service = HuggingFaceService()
        logger.info("AI service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Gen-AI Text Generation API",
    description="AI-Powered Text Generation API using GPT-2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict, tags=["Root"])
async def root():
    return {
        "message": "Welcome to Gen-AI Text Generation API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "generate": "POST /generate - Generate text from a prompt",
            "health": "GET /health - Check API health"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=ai_service is not None
    )

@app.post("/generate", response_model=GenerationResponse, status_code=status.HTTP_200_OK, tags=["Generation"])
async def generate_text(request: GenerationRequest):
    if not ai_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service not initialized"
        )
    
    try:
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        result = ai_service.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        logger.info(f"Generated text for prompt: {request.prompt[:50]}...")
        return GenerationResponse(**result)
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )