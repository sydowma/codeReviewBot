import argparse
import hashlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from threading import Thread
from time import sleep
from typing import Union

import gitlab
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from github import Github, logger
from github.PullRequest import PullRequest
from openai import OpenAI
from pydantic import BaseModel

# 最后一次拉取的PR编号，服务启动后设置默认值
LATEST_PULL_REQUEST_NUMBER: int = 0

def background_task():
    while True:
        # get github latest pull request
        git = GitHubReview()
        git.review_last_pull_request()
        print("执行定时任务...")
        sleep(10)

# 创建 Lifespan 管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("应用启动前的初始化...")
    # 启动定时任务线程
    task = Thread(target=background_task)
    task.daemon = True
    task.start()
    # 可以在这里初始化数据库连接、缓存、队列等资源
    yield
    print("应用关闭后的清理...")
    # 可以在这里关闭数据库连接、清理缓存等

app = FastAPI(lifespan=lifespan)

# 配置文件的默认路径
DEFAULT_CONFIG_PATH = 'config.json'
# 环境变量文件的默认路径
DEFAULT_ENV_FILE = '.env'

def load_config(config_path=DEFAULT_CONFIG_PATH, env_file=DEFAULT_ENV_FILE):
    # 加载 .env 文件中的环境变量
    load_dotenv(env_file)

    # 默认配置
    config = {
        "GITLAB_URL": "",
        "GITLAB_TOKEN": "",
        "GITHUB_REPO_URL": "",
        "GITHUB_TOKEN": "",
        "OPENAI_API_KEY": "",
        "OPENAI_HTTP_PROXY": "",
        "OLLAMA_URL": "http://localhost:11434",  # Default Ollama URL
        "AI_PROVIDER": "openai",  # Default AI provider
        "MODEL": "gpt-4o",  # Default model
    }

    # 如果配置文件存在，从文件中读取配置
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)

    # 环境变量覆盖配置文件中的值
    for key in config:
        env_value = os.getenv(key)
        if env_value:
            config[key] = env_value

    return config

# 使用示例
config = load_config()
GITLAB_URL = config['GITLAB_URL']
GITLAB_TOKEN = config['GITLAB_TOKEN']
GITHUB_TOKEN = config['GITHUB_TOKEN']
OPENAI_API_KEY = config['OPENAI_API_KEY']
OPENAI_HTTP_PROXY = config['OPENAI_HTTP_PROXY']
OLLAMA_URL = config['OLLAMA_URL']
AI_PROVIDER = config['AI_PROVIDER']
MODEL = config['MODEL']

# Prompt file paths
DETAILED_PROMPT_FILE = "detailed_prompt.txt"
SUMMARY_PROMPT_FILE = "summary_prompt.txt"

class ReviewResult:
    def __init__(self):
        self.comments: list[ReviewComment] = []
        self.summary = ""

class AIClient(ABC):
    @abstractmethod
    def get_client(self):
        pass

class OpenAIClient(AIClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http_proxy = OPENAI_HTTP_PROXY

    def get_client(self) -> OpenAI:
        if len(OPENAI_HTTP_PROXY) > 0:
            http_client = httpx.Client(proxies={"http://": self.http_proxy, "https://": self.http_proxy})
            return OpenAI(api_key=self.api_key, http_client=http_client)
        return OpenAI(api_key=self.api_key)

class OllamaClient(AIClient):
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout

    def get_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

class AIClientFactory:
    @staticmethod
    def create_client(provider: str, **kwargs) -> AIClient:
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "ollama":
            return OllamaClient(**kwargs)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")

class IssueByLineNumber(BaseModel):
    comment: str
    line: int

class FileIssues(BaseModel):
    file_path: str
    issues: list[IssueByLineNumber]

class IssuesComment(BaseModel):
    summary: str
    file_issues: list[FileIssues]

class ReviewComment(BaseModel):
    file: str
    line: int
    comment: str

class BaseReview(ABC):
    def __init__(self):
        self.detailed_prompt = self.read_prompt(DETAILED_PROMPT_FILE)
        self.summary_prompt = self.read_prompt(SUMMARY_PROMPT_FILE)
        self.ai_provider = AI_PROVIDER
        self.model = MODEL
        self.ai_client: AIClient = AIClientFactory.create_client(AI_PROVIDER, api_key=OPENAI_API_KEY)
        self.client: Union[OpenAI, httpx.Client] = self.ai_client.get_client()

    def read_prompt(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file {file_path} not found. Using default prompt.")
            return ""

    def call_ai_api(self, review_request, summary_only=False):
        try:
            if self.ai_provider == "openai":
                if summary_only is True:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system",
                             "content": "You are an expert code reviewer with programmer. Provide detailed, feedback on the code changes."},
                            {"role": "user", "content": review_request}
                        ]
                    )
                    return response.choices[0].message.content

                if summary_only is False:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system",
                            "content": "You are an expert code reviewer with programmer. Provide detailed, line-specific feedback on the code changes."},
                            {"role": "user", "content": review_request}
                        ],
                        response_format=IssuesComment
                    )
                    return response.choices[0].message.parsed
            elif self.ai_provider == "ollama":
                response = self.client.post("/api/generate", json={
                    "model": self.model,
                    "prompt": review_request,
                    "stream": False
                })
                return response.json()['response']
            else:
                raise ValueError(f"Unsupported AI provider: {self.ai_provider}")
        except Exception as e:
            print(f"Error calling AI API: {e}", file=sys.stderr)
            return "Error: Unable to complete code review due to API issues."

    def parse_review_result(self, review_result: Union[str, IssuesComment], summary_only: bool) -> ReviewResult:
        result = ReviewResult()
        if summary_only:
            result.summary = review_result
        else:
            print(review_result)
            issues_comment = review_result
            result.summary = issues_comment.summary
            for file_issue in issues_comment.file_issues:
                for issue in file_issue.issues:
                    result.comments.append(
                        ReviewComment(file=file_issue.file_path, line=issue.line, comment=issue.comment)
                    )

        return result

    @abstractmethod
    def review_code_changes(self, url: str, summary_only: bool = False):
        pass

    @abstractmethod
    def parse_url(self, url):
        pass

    @abstractmethod
    def build_review_request(self, changes: object, summary_only: bool, mr_information: str) -> object:
        pass

    @abstractmethod
    def submit_comments(self, code_change, comments, changes):
        pass

    @abstractmethod
    def create_position(self, code_change, changes, file_path, line_number):
        pass

    def find_line_numbers(self, diff, target_line):
        """
                Parse diff and find the correct line numbers for commenting
                """
        lines = diff.split('\n')
        current_line = 0
        in_added_lines = False
        old_line = 0
        new_line = 0

        for line in lines:
            if line.startswith('@@'):
                # Parse the hunk header
                match = re.match(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    old_line = int(match.group(1))
                    new_line = int(match.group(2))
                    current_line = 0
                    continue

            if line.startswith('-'):
                old_line += 1
                in_added_lines = False
            elif line.startswith('+'):
                new_line += 1
                current_line += 1
                in_added_lines = True
            else:
                old_line += 1
                new_line += 1
                current_line += 1
                in_added_lines = False

            if current_line == target_line:
                # For added lines, we want the new line number
                if in_added_lines:
                    return old_line - 1, new_line
                # For context lines, we want both old and new line numbers
                return old_line, new_line

        return None, None

    def generate_line_code(self, file_path, old_line, new_line):
        file_hash = hashlib.sha1(file_path.encode()).hexdigest()
        return f"{file_hash}_{old_line}_{new_line}"

class GitLabReview(BaseReview):
    def __init__(self):
        super().__init__()
        self.gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)

    def review_code_changes(self, merge_request_url: str, summary_only: bool):
        try:
            project_id, merge_request_iid = self.parse_url(merge_request_url)
            project = self.gl.projects.get(project_id)
            mr = project.mergerequests.get(merge_request_iid)
            changes = mr.changes()
            mr_title = mr.title
            mr_description = mr.description

            mr_infomation = self.get_body(mr_description, mr_title)

            review_request = self.build_review_request(changes, summary_only, mr_infomation)
            review_result = self.call_ai_api(review_request)
            parsed_result = self.parse_review_result(review_result, summary_only)

            if summary_only:
                if parsed_result.summary:
                    mr.notes.create({'body': parsed_result.summary})
            else:
                self.submit_comments(mr, parsed_result.comments, changes)
                if parsed_result.summary:
                    mr.notes.create({'body': parsed_result.summary})

            return {"status": "success", "message": "Code review completed and comments posted to merge request"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_body(self, mr_description, mr_title):
        mr_infomation = f"Merge Request Title: {mr_title}\n Merge Request Description: {mr_description}\n\n"
        return mr_infomation

    def parse_url(self, url):
        parts = url.split('/')
        project_id = parts[-5] + '/' + parts[-4]
        merge_request_iid = parts[-1]
        return project_id, merge_request_iid

    def build_review_request(self, changes, summary_only, mr_infomation: str):
        files_content = []
        for change in changes['changes']:
            is_update: bool = change['new_path'] == change['old_path']
            if is_update:
                change_type = 'Update'
            else:
                change_type = 'Add' if change['old_path'] == '/dev/null' else 'Delete'

            files_content.append(f"Change type is {change_type} File: {change['new_path']}\n\n{change['diff']}")

        prompt = self.summary_prompt if summary_only else self.detailed_prompt
        return prompt + "\n\n" + mr_infomation + "\n\n".join(files_content)

    def submit_comments(self, pr: PullRequest, comments: list[ReviewComment], changes):
        """
        Submit review comments to the pull request with proper positioning
        """
        try:
            # Start a new review
            review_comments = []

            for comment in comments:
                try:
                    position = self.create_position(pr, changes, comment.file, comment.line)
                    if position:
                        review_comments.append({
                            'path': position['path'],
                            'position': position['position'],
                            'body': comment.comment,
                            'line': position['line'],
                            'side': position['side']
                        })
                    else:
                        print(f"Warning: Could not create position for file {comment.file} line {comment.line}")
                except Exception as e:
                    print(f"Error creating comment position: {str(e)}")
                    continue

            if review_comments:
                # Submit all comments as a single review
                head_commit = pr.get_commits().reversed[0]
                pr.create_review(
                    commit=head_commit,
                    comments=review_comments,
                    event='COMMENT'
                )
        except Exception as e:
            print(f"Failed to submit review comments: {str(e)}")

    def create_position(self, pr: PullRequest, changes, file_path: str, target_line: int):
        """
        Create the correct position object for GitHub review comments
        """
        for change in changes['changes']:
            if change['new_path'] == file_path:
                old_line, new_line = self.find_line_numbers(change['diff'], target_line)

                if old_line is not None and new_line is not None:
                    # For added/modified lines
                    return {
                        'path': file_path,
                        'position': new_line,
                        'line': new_line,
                        'side': 'RIGHT'
                    }
                elif old_line is not None:
                    # For deleted lines
                    return {
                        'path': file_path,
                        'position': old_line,
                        'line': old_line,
                        'side': 'LEFT'
                    }
        return None

class GitHubReview(BaseReview):
    def __init__(self):
        super().__init__()
        self.gh = Github(GITHUB_TOKEN)

    def review_code_changes(self, pull_request_url: str, summary_only: bool):
        try:
            repo_full_name, pull_request_number = self.parse_url(pull_request_url)
            repo = self.gh.get_repo(repo_full_name)
            pr = repo.get_pull(pull_request_number)
            changes = self.get_pull_request_changes(pr)
            body: str = self.get_body(pr)
            review_request = self.build_review_request(changes, summary_only, body)
            review_result = self.call_ai_api(review_request, summary_only)
            parsed_result = self.parse_review_result(review_result, summary_only)

            if summary_only:
                if parsed_result.summary:
                    pr.create_issue_comment(parsed_result.summary)
            else:
                self.submit_comments(pr, parsed_result.comments, changes)
                if parsed_result.summary:
                    pr.create_issue_comment(parsed_result.summary)

            return {"status": "success", "message": "Code review completed and comments posted to pull request"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def parse_url(self, url):
        parts = url.split('/')
        repo_full_name = '/'.join(parts[-4:-2])
        pull_request_number = int(parts[-1])
        return repo_full_name, pull_request_number

    def get_pull_request_changes(self, pr: PullRequest):
        """
        Get changes with improved diff handling
        """
        files = pr.get_files()
        changes = {'changes': []}

        for file in files:
            if file.patch:  # Only include files with actual changes
                change = {
                    'new_path': file.filename,
                    'old_path': file.previous_filename or file.filename,
                    'diff': file.patch,
                    'status': file.status,  # Added, Modified, Removed
                    'additions': file.additions,
                    'deletions': file.deletions,
                    'changes': file.changes
                }
                changes['changes'].append(change)

        return changes

    def build_review_request(self, changes, summary_only: bool, mr_infomation: str):
        files_content = []
        for change in changes['changes']:
            files_content.append(f"File: {change['new_path']}\n\n{change['diff']}")

        prompt = self.summary_prompt if summary_only else self.detailed_prompt
        return prompt + "\n\n" + mr_infomation + "\n\n" + "\n\n".join(files_content)

    def submit_comments(self, pr, comments: list[ReviewComment], changes):
        for comment in comments:
            try:
                position = self.create_position(pr, changes, comment.file, comment.line)
                if position:
                    head_commit = pr.get_commits().reversed[0]
                    pr_comment = pr.create_review_comment(
                        body=comment.comment,
                        path=position['path'],
                        line=position['line'],
                        side=position['side'],
                        commit=head_commit
                    )
                    print(f"Created comment: {pr_comment}")
                else:
                    print(
                        f"Warning: Could not create position for file {comment.file} line {comment.line}. Skipping comment.")
            except Exception as e:
                print(f"Failed to create comment. Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(
                    f"Comment details: File: {comment.file}, Line: {comment.line}, Comment: {comment.comment[:50]}...")

    def create_position(self, pr: PullRequest, changes, file_path: str, target_line: int):
        """
        Create the correct position object for GitHub review comments
        """
        for change in changes['changes']:
            if change['new_path'] == file_path:
                old_line, new_line = self.find_line_numbers(change['diff'], target_line)

                if old_line is not None and new_line is not None:
                    # For added/modified lines
                    return {
                        'path': file_path,
                        'position': new_line,
                        'line': new_line,
                        'side': 'RIGHT'
                    }
                elif old_line is not None:
                    # For deleted lines
                    return {
                        'path': file_path,
                        'position': old_line,
                        'line': old_line,
                        'side': 'LEFT'
                    }
        return None

    def get_body(self, pr: PullRequest) -> str:
        return f"Pull Request Title: {pr.title}\n Pull Request Description: {pr.body}\n\n"

    def review_last_pull_request(self):
        global LATEST_PULL_REQUEST_NUMBER
        pull_requests = self.gh.get_repo(config['GITHUB_REPO_URL']).get_pulls(state='open', sort='created', direction='desc')
        if LATEST_PULL_REQUEST_NUMBER == 0:
            LATEST_PULL_REQUEST_NUMBER = pull_requests[0].number
            logger.info(f"Set latest pull request number to {LATEST_PULL_REQUEST_NUMBER}")
            return
        for pr in pull_requests:
            if pr.number > LATEST_PULL_REQUEST_NUMBER:
                self.review_code_changes(pr.url, False)
                LATEST_PULL_REQUEST_NUMBER = pr.number
                logger.info(f"Reviewed pull request #{pr.number}")


class CodeChangeInput(BaseModel):
    url: str
    summary_only: bool = False
    ai_provider: str = "openai"
    model: str = "gpt-4o"

@app.post("/review")
async def api_review_code_changes(input: CodeChangeInput):
    if "gitlab" in input.url:
        review = GitLabReview()
    elif "github" in input.url:
        review = GitHubReview()
    else:
        raise HTTPException(status_code=400, detail="Unsupported repository type")

    result = review.review_code_changes(input.url, input.summary_only)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

def cli():
    parser = argparse.ArgumentParser(description="Code Review CLI")
    parser.add_argument("url", help="URL of the GitLab merge request or GitHub pull request to review")
    parser.add_argument("--summary", action="store_true", default=True, help="Generate a summary review")
    parser.add_argument("--no-summary", action="store_false", dest="summary", help="Do not generate a summary review")
    args = parser.parse_args()

    if "gitlab" in args.url:
        review = GitLabReview()
    elif "github" in args.url:
        review = GitHubReview()
    else:
        print("Error: Unsupported repository type", file=sys.stderr)
        sys.exit(1)

    # Update AI provider and model based on CLI arguments
    review.ai_provider = AI_PROVIDER
    review.model = MODEL

    result = review.review_code_changes(args.url, args.summary)
    if result["status"] == "success":
        print(result["message"])
    else:
        print(f"Error: {result['message']}", file=sys.stderr)
        sys.exit(1)




if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)