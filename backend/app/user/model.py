from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    email: str = Field(unique=True, index=True)
    username: str = Field(unique=True)
    github_access_token: Optional[str] = Field(default=None)
    github_refresh_token: Optional[str] = Field(default=None)
    gitlab_access_token: Optional[str] = Field(default=None)
    gitlab_refresh_token: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    repositories: List["Repository"] = Relationship(back_populates="owner")
    reviews: List["Review"] = Relationship(back_populates="reviewer")