import hashlib
import os
import sys
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import gitlab
from openai import OpenAI
import uvicorn
import httpx
import re

app = FastAPI()

# 从环境变量中读取配置
GITLAB_URL = os.getenv("GITLAB_URL")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN") or ''
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ''
OPENAI_HTTP_PROXY = os.getenv("OPENAI_HTTP_PROXY") or ''

# Prompt 文件路径
DETAILED_PROMPT_FILE = "detailed_prompt.txt"
SUMMARY_PROMPT_FILE = "summary_prompt.txt"

class ReviewResult:
    def __init__(self):
        self.comments = []
        self.summary = ""

class Review(object):
    def __init__(self):
        # 初始化GitLab客户端
        self.gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)

        # 初始化OpenAI客户端
        if len(OPENAI_HTTP_PROXY) > 0:
            http_client = httpx.Client(proxies={"http://": OPENAI_HTTP_PROXY, "https://": OPENAI_HTTP_PROXY})
            self.client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)

            # 读取 prompts
        self.detailed_prompt = self.read_prompt(DETAILED_PROMPT_FILE)
        self.summary_prompt = self.read_prompt(SUMMARY_PROMPT_FILE)

    def read_prompt(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file {file_path} not found. Using default prompt.")
            return ""

    def review_merge_request(self, merge_request_url: str, summary_only: bool = False):
        try:
            # 解析merge request URL
            project_id, merge_request_iid = self.parse_merge_request_url(merge_request_url)

            # 获取merge request详情
            project = self.gl.projects.get(project_id)
            mr = project.mergerequests.get(merge_request_iid)

            # 获取merge request的变更内容
            changes = mr.changes()

            # 构建code review请求
            review_request = self.build_review_request(changes, summary_only)

            # 调用OpenAI API进行code review
            review_result = self.call_openai_api(review_request)

            parsed_result = self.parse_review_result(review_result, summary_only)
            print(parsed_result)

            if summary_only:
                # 只提交摘要评论
                if parsed_result.summary:
                    mr.notes.create({'body': parsed_result.summary})
            else:
                # 提交行级评论和摘要评论
                self.submit_comments(mr, parsed_result.comments, changes)
                if parsed_result.summary:
                    mr.notes.create({'body': parsed_result.summary})

            return {"status": "success", "message": "Code review completed and comments posted to merge request"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def parse_merge_request_url(self, url):
        parts = url.split('/')
        project_id = parts[-5] + '/' + parts[-4]
        merge_request_iid = parts[-1]
        return project_id, merge_request_iid

    def build_review_request(self, changes, summary_only):
        files_content = []
        for change in changes['changes']:
            files_content.append(f"File: {change['new_path']}\n\n{change['diff']}")

        prompt = self.summary_prompt if summary_only else self.detailed_prompt
        return prompt + "\n\n" + "\n\n".join(files_content)


    def call_openai_api(self, review_request):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert code reviewer. Provide detailed, line-specific feedback on the code changes."},
                    {"role": "user", "content": review_request}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}", file=sys.stderr)
            return "Error: Unable to complete code review due to API issues."

    def find_line_code(self, changes, file_path, line_number):
        for change in changes['changes']:
            if change['new_path'] == file_path:
                diff_lines = change['diff'].split('\n')
                current_line = 0
                for diff_line in diff_lines:
                    if diff_line.startswith('+') and not diff_line.startswith('+++'):
                        current_line += 1
                        if current_line == line_number:
                            # 构造line_code
                            return f"{change['new_path']}_{current_line}"
        return None

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
                    print(
                        f"Warning: Could not create position for file {comment['file']} line {comment['line']}. Skipping comment.")
            except gitlab.exceptions.GitlabCreateError as e:
                print(f"Failed to create comment: {e}")
                print(
                    f"Comment details: File: {comment['file']}, Line: {comment['line']}, Comment: {comment['comment'][:50]}...")

    def parse_review_result(self, review_result: str, summary_only: bool):
        result = ReviewResult()
        if summary_only:
            result.summary = review_result
        else:
            # 详细解析逻辑保持不变
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


class MergeRequestInput(BaseModel):
    merge_request_url: str
    summary_only: bool = False


@app.post("/review")
async def api_review_merge_request(input: MergeRequestInput):
    review = Review()
    result = review.review_merge_request(input.merge_request_url)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


def cli():
    parser = argparse.ArgumentParser(description="GitLab Merge Request Code Review CLI")
    parser.add_argument("merge_request_url", help="URL of the GitLab merge request to review")
    args = parser.parse_args()

    review = Review()
    result = review.review_merge_request(args.merge_request_url)
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
