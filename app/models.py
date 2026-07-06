from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text to generate from", min_length=1)
    max_length: int = Field(100, description="Maximum length of generated text", ge=1, le=500)
    temperature: float = Field(0.7, description="Sampling temperature (higher = more random)", ge=0.1, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Once upon a time",
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    model: str
    parameters: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "Once upon a time in a faraway land...",
                "model": "gpt2",
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool