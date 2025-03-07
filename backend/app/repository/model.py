from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
import json
from ..user.model import User  # Import User from its new location

class Repository(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    owner_id: int = Field(foreign_key="user.id")
    name: str
    platform: str = Field(regex="^(github|gitlab)$")
    api_token: str
    custom_prompt: Optional[str] = Field(default=None)
    settings: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    owner: User = Relationship(back_populates="repositories")
    reviews: List["Review"] = Relationship(back_populates="repository")

    def get_settings(self) -> dict:
        return json.loads(self.settings)
    
    def set_settings(self, settings: dict):
        self.settings = json.dumps(settings)