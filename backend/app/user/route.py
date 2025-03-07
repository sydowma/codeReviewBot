from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from ..dependencies import get_session
from .model import User
from .schema import UserCreate, UserResponse

router = APIRouter()

@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, session: Session = Depends(get_session)):
    db_user = User.from_orm(user)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user