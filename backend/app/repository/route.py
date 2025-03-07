from fastapi import APIRouter, Depends
from sqlmodel import Session
from ..dependencies import get_session
from .model import Repository
from .schema import RepositoryCreate, RepositoryResponse

router = APIRouter()

@router.post("/", response_model=RepositoryResponse)
def create_repository(repo: RepositoryCreate, session: Session = Depends(get_session)):
    db_repo = Repository.from_orm(repo, update={"owner_id": 1})  # Hardcoded owner_id for now
    db_repo.set_settings(repo.settings.dict())
    session.add(db_repo)
    session.commit()
    session.refresh(db_repo)
    return db_repo