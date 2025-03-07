from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
    GITLAB_CLIENT_ID = os.getenv("GITLAB_CLIENT_ID")
    GITLAB_CLIENT_SECRET = os.getenv("GITLAB_CLIENT_SECRET")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database.db")

config = Config()