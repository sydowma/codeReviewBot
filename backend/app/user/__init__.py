# app/user/__init__.py
from .model import User
from .schema import UserCreate, UserResponse
from .route import router as user_router