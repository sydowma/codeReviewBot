import hashlib
import json
import os
import sys
import argparse
from contextlib import asynccontextmanager
from datetime import time
from sched import scheduler
from threading import Thread
from time import sleep

from charset_normalizer.constant import LANGUAGE_SUPPORTED_COUNT
from fastapi import FastAPI, HTTPException
from github.PullRequest import PullRequest
from pydantic import BaseModel
import requests
import gitlab
from github import Github, logger
from openai import OpenAI
import uvicorn
import httpx
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from pygments.lexer import default
from fastapi import BackgroundTasks, FastAPI

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
        self.comments = []
        self.summary = ""

class BaseReview(ABC):
    def __init__(self):
        self.detailed_prompt = self.read_prompt(DETAILED_PROMPT_FILE)
        self.summary_prompt = self.read_prompt(SUMMARY_PROMPT_FILE)
        self.ai_provider = AI_PROVIDER
        self.model = MODEL

        # Initialize OpenAI client
        if self.ai_provider == "openai":
            if len(OPENAI_HTTP_PROXY) > 0:
                http_client = httpx.Client(proxies={"http://": OPENAI_HTTP_PROXY, "https://": OPENAI_HTTP_PROXY})
                self.client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            else:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.ai_provider == "ollama":
            self.client = httpx.Client(base_url=OLLAMA_URL, timeout=120)

    def read_prompt(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file {file_path} not found. Using default prompt.")
            return ""

    def call_ai_api(self, review_request):
        try:
            if self.ai_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert code reviewer. Provide detailed, line-specific feedback on the code changes."},
                        {"role": "user", "content": review_request}
                    ]
                )
                return response.choices[0].message.content
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

    def parse_review_result(self, review_result: str, summary_only: bool):
        result = ReviewResult()
        if summary_only:
            result.summary = review_result
        else:
            current_file = None
            current_lines = None
            current_comment = []
            summary_started = False

            for line in review_result.split('\n'):
                line = line.strip()
                if line.startswith('FILE:'):
                    if current_file and current_lines is not None and current_comment:
                        result.comments.append({
                            'file': current_file,
                            'line': current_lines,
                            'comment': '\n'.join(current_comment)
                        })
                    current_file = line.split(':')[1].strip()
                    current_lines = None
                    current_comment = []
                elif line.startswith('LINES:') or line.startswith('LINE:'):
                    if current_file and current_lines is not None and current_comment:
                        result.comments.append({
                            'file': current_file,
                            'line': current_lines,
                            'comment': '\n'.join(current_comment)
                        })
                    current_lines = line.split(':')[1].strip()
                    current_lines = int(current_lines.split('-')[-1])  # Get the last number
                    current_comment = []
                elif line.startswith('General Comments:'):
                    summary_started = True
                    if current_file and current_lines is not None and current_comment:
                        result.comments.append({
                            'file': current_file,
                            'line': current_lines,
                            'comment': '\n'.join(current_comment)
                        })
                    current_file = None
                    current_lines = None
                    current_comment = []
                elif summary_started:
                    result.summary += line + '\n'
                elif current_file is not None:
                    current_comment.append(line)

            if current_file and current_lines is not None and current_comment:
                result.comments.append({
                    'file': current_file,
                    'line': current_lines,
                    'comment': '\n'.join(current_comment)
                })

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
        lines = diff.split('\n')
        old_line = new_line = 0
        for line in lines:
            if line.startswith('+'):
                new_line += 1
                if new_line == target_line:
                    return old_line, new_line
            elif line.startswith('-'):
                old_line += 1
            else:
                old_line += 1
                new_line += 1
                if new_line == target_line:
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

    def submit_comments(self, mr, comments, changes):
        for comment in comments:
            try:
                position = self.create_position(mr, changes, comment['file'], comment['line'])
                if position:
                    mr.discussions.create({
                        'body': comment['comment'],
                        'position': position
                    })
                else:
                    print(f"Warning: Could not create position for file {comment['file']} line {comment['line']}. Skipping comment.")
            except gitlab.exceptions.GitlabCreateError as e:
                print(f"Failed to create comment: {e}")
                print(f"Comment details: File: {comment['file']}, Line: {comment['line']}, Comment: {comment['comment'][:50]}...")

    def create_position(self, mr, changes, file_path, line_number):
        for change in changes['changes']:
            if change['new_path'] == file_path:
                old_line, new_line = self.find_line_numbers(change['diff'], line_number)
                if old_line is not None and new_line is not None:
                    return {
                        'base_sha': mr.diff_refs['base_sha'],
                        'start_sha': mr.diff_refs['start_sha'],
                        'head_sha': mr.diff_refs['head_sha'],
                        'position_type': 'text',
                        'new_path': file_path,
                        'new_line': new_line,
                        'old_path': change['old_path'],
                        'old_line': old_line,
                        'line_range': {
                            'start': {
                                'line_code': self.generate_line_code(file_path, old_line, new_line),
                                'type': 'new'
                            },
                            'end': {
                                'line_code': self.generate_line_code(file_path, old_line, new_line),
                                'type': 'new'
                            }
                        }
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
            review_result = self.call_ai_api(review_request)
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

    def get_pull_request_changes(self, pr):
        files = pr.get_files()
        changes = {'changes': []}
        for file in files:
            changes['changes'].append({
                'new_path': file.filename,
                'old_path': file.previous_filename or file.filename,
                'diff': file.patch
            })
        return changes

    def build_review_request(self, changes, summary_only: bool, mr_infomation: str):
        files_content = []
        for change in changes['changes']:
            files_content.append(f"File: {change['new_path']}\n\n{change['diff']}")

        prompt = self.summary_prompt if summary_only else self.detailed_prompt
        return prompt + "\n\n" + mr_infomation + "\n\n" + "\n\n".join(files_content)

    def submit_comments(self, pr, comments, changes):
        for comment in comments:
            try:
                position = self.create_position(pr, changes, comment['file'], comment['line'])
                if position:
                    pr.create_review_comment(
                        body=comment['comment'],
                        path=position['path'],
                        position=position['position'],
                        commit_id=pr.head.sha
                    )
                else:
                    print(f"Warning: Could not create position for file {comment['file']} line {comment['line']}. Skipping comment.")
            except Exception as e:
                print(f"Failed to create comment: {e}")
                print(f"Comment details: File: {comment['file']}, Line: {comment['line']}, Comment: {comment['comment'][:50]}...")

    def create_position(self, pr, changes, file_path, line_number):
        for change in changes['changes']:
            if change['new_path'] == file_path:
                old_line, new_line = self.find_line_numbers(change['diff'], line_number)
                if old_line is not None and new_line is not None:
                    return {
                        'path': file_path,
                        'position': new_line
                    }
        return None

    def get_body(self, pr: PullRequest) -> str:
        return f"Pull Request Title: {pr.title}\n Pull Request Description: {pr.body}\n\n"

    def review_last_pull_request(self):
        global LATEST_PULL_REQUEST_NUMBER
        pull_requests = self.gh.get_repo(GITLAB_URL).get_pulls(state='open', sort='created', direction='desc')
        if LATEST_PULL_REQUEST_NUMBER == 0:
            LATEST_PULL_REQUEST_NUMBER = pull_requests[0].number
            logger.info(f"Set latest pull request number to {LATEST_PULL_REQUEST_NUMBER}")
            return
        for pr in pull_requests:
            if pr.number > LATEST_PULL_REQUEST_NUMBER:
                changes = self.get_pull_request_changes(pr)
                body: str = self.get_body(pr)
                review_request = self.build_review_request(changes, False, body)
                review_result = self.call_ai_api(review_request)
                parsed_result = self.parse_review_result(review_result, False)
                self.submit_comments(pr, parsed_result.comments, changes)
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
    parser.add_argument("--summary", action="store_true", help="Generate only a summary review", default=True)
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