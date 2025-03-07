from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel as SQLModelBase

class ReviewCommentCreate(SQLModelBase):
    file_path: str
    line_number: int
    comment: str

class ReviewCommentResponse(SQLModelBase):
    id: int
    review_id: int
    file_path: str
    line_number: int
    comment: str
    created_at: datetime
    
    class Config:
        orm_mode = True