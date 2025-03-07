from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import datetime
from ..review.model import Review

class ReviewComment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    review_id: int = Field(foreign_key="review.id")
    file_path: str
    line_number: int
    comment: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    review: Review = Relationship(back_populates="comments")