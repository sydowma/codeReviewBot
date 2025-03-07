from fastapi import APIRouter, Depends
from sqlmodel import Session
from ..dependencies import get_session
from .model import Review
from .schema import ReviewCreate, ReviewResponse

router = APIRouter()

@router.post("/", response_model=ReviewResponse)
def create_review(review: ReviewCreate, session: Session = Depends(get_session)):
    db_review = Review.from_orm(review)
    session.add(db_review)
    session.commit()
    session.refresh(db_review)
    return db_review