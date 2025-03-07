from fastapi import APIRouter, Depends
from sqlmodel import Session
from ..dependencies import get_session
from .model import ReviewComment
from .schema import ReviewCommentCreate, ReviewCommentResponse

router = APIRouter()

@router.post("/", response_model=ReviewCommentResponse)
def create_review_comment(comment: ReviewCommentCreate, session: Session = Depends(get_session)):
    db_comment = ReviewComment.from_orm(comment)
    session.add(db_comment)
    session.commit()
    session.refresh(db_comment)
    return db_comment