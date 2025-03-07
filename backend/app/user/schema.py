from pydantic import EmailStr
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel as SQLModelBase

class UserCreate(SQLModelBase):
    email: EmailStr
    username: str

class UserResponse(SQLModelBase):
    id: int
    email: EmailStr
    username: str
    created_at: datetime
    
    class Config:
        orm_mode = True