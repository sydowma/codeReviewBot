from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from ..config import config

oauth = OAuth(Config(".env"))

oauth.register(
    name="github",
    client_id=config.GITHUB_CLIENT_ID,
    client_secret=config.GITHUB_CLIENT_SECRET,
    authorize_url="https://github.com/login/oauth/authorize",
    access_token_url="https://github.com/login/oauth/access_token",
    api_base_url="https://api.github.com/",
)

oauth.register(
    name="gitlab",
    client_id=config.GITLAB_CLIENT_ID,
    client_secret=config.GITLAB_CLIENT_SECRET,
    authorize_url="https://gitlab.com/oauth/authorize",
    access_token_url="https://gitlab.com/oauth/token",
    api_base_url="https://gitlab.com/api/v4/",
)