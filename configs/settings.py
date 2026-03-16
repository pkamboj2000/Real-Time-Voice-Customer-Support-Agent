"""
Application settings — loaded from .env at startup.

We use pydantic-settings so env vars are validated eagerly and we get
nice error messages when something is missing instead of a random
KeyError deep in the call graph.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    """Central config pulled from environment variables."""

    # --- OpenAI ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # --- Deepgram ---
    deepgram_api_key: str = Field(default="")

    # --- ElevenLabs ---
    elevenlabs_api_key: str = Field(default="")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM")

    # --- Twilio ---
    twilio_account_sid: str = Field(default="")
    twilio_auth_token: str = Field(default="")
    twilio_phone_number: str = Field(default="")

    # --- Vector Store ---
    vector_store_type: str = Field(default="faiss")
    vector_store_path: str = Field(default="./data/vector_index")
    embedding_dimension: int = Field(default=1536)

    # --- App ---
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    # --- Agent tuning ---
    confidence_threshold: float = Field(default=0.65)
    max_retrieval_results: int = Field(default=5)
    escalation_keywords: str = Field(default="manager,human,supervisor,representative,person")

    # --- Audio ---
    audio_sample_rate: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    audio_chunk_ms: int = Field(default=100)

    @property
    def escalation_keyword_list(self) -> List[str]:
        """Split the comma-separated keywords into a proper list."""
        return [kw.strip().lower() for kw in self.escalation_keywords.split(",")]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# singleton — import this wherever you need config
settings = Settings()
