from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "Gen-AI Text Generation API"
    app_version: str = "1.0.0"
    debug: bool = False
    model_name: str = "gpt2"
    max_length_limit: int = 500
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()