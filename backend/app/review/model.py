from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from ..user.model import User
from ..repository.model import Repository

class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    repository_id: int = Field(foreign_key="repository.id")
    reviewer_id: Optional[int] = Field(default=None, foreign_key="user.id")
    pr_or_mr_id: str
    title: str
    summary: Optional[str] = Field(default=None)
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    repository: Repository = Relationship(back_populates="reviews")
    reviewer: Optional[User] = Relationship(back_populates="reviews")
    comments: List["ReviewComment"] = Relationship(back_populates="review")