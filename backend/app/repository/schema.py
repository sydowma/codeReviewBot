from pydantic import Field
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel as SQLModelBase

class RepositorySettings(SQLModelBase):
    ai_provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-4"

class RepositoryCreate(SQLModelBase):
    name: str
    platform: str = Field(..., regex="^(github|gitlab)$")
    api_token: str
    custom_prompt: Optional[str] = None
    settings: Optional[RepositorySettings] = Field(default_factory=lambda: RepositorySettings())

class RepositoryResponse(SQLModelBase):
    id: int
    owner_id: int
    name: str
    platform: str
    custom_prompt: Optional[str]
    settings: RepositorySettings
    created_at: datetime
    
    class Config:
        orm_mode = True
        property_getters = {"settings": lambda obj: obj.get_settings()}