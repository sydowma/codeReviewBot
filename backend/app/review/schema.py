from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel as SQLModelBase
from ..review_comment.schema import ReviewCommentResponse

class ReviewCreate(SQLModelBase):
    repository_id: int
    pr_or_mr_id: str
    title: str

class ReviewResponse(SQLModelBase):
    id: int
    repository_id: int
    reviewer_id: Optional[int]
    pr_or_mr_id: str
    title: str
    summary: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    comments: List[ReviewCommentResponse] = []
    
    class Config:
        orm_mode = True