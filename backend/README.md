# Code Review Tool

This tool automates code reviews for GitLab merge requests and GitHub pull requests using OpenAI's GPT model.

## Features

- Supports both GitLab and GitHub repositories
- Generates detailed, line-specific feedback on code changes
- Provides summary reviews
- Can be used as a CLI tool or run as a FastAPI server

## Requirements

- Python 3.7+
- FastAPI
- GitPython
- PyGitHub
- python-gitlab
- OpenAI Python Client
- python-dotenv
- uvicorn (for running the FastAPI server)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sydowma/codeReviewBot.git
   cd codeReviewBot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration:
   - Create a `config.json` file with your GitLab, GitHub, and OpenAI API credentials.
   - Alternatively, set up environment variables in a `.env` file.

## Usage

### As a CLI tool

```
python main.py <url_to_merge_request_or_pull_request> [--summary-only]
```

### As a FastAPI server

1. Start the server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Send a POST request to `http://localhost:8000/review` with the following JSON body:
   ```json
   {
     "url": "https://gitlab.com/your-project/merge_requests/1",
     "summary_only": false
   }
   ```

## Configuration

The tool uses the following configuration options:

- `GITLAB_URL`: Your GitLab instance URL
- `GITLAB_TOKEN`: Your GitLab personal access token
- `GITHUB_TOKEN`: Your GitHub personal access token
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_HTTP_PROXY`: (Optional) HTTP proxy for OpenAI API calls

You can set these in `config.json` or as environment variables.

## Customization

You can customize the review prompts by modifying the `detailed_prompt.txt` and `summary_prompt.txt` files.

---

# 代码审查工具

这个工具使用OpenAI的GPT模型自动化GitLab合并请求和GitHub拉取请求的代码审查。

## 特性

- 支持GitLab和GitHub仓库
- 生成详细的、针对具体行的代码变更反馈
- 提供摘要审查
- 可以作为CLI工具使用，也可以作为FastAPI服务器运行

## 要求

- Python 3.7+
- FastAPI
- GitPython
- PyGitHub
- python-gitlab
- OpenAI Python客户端
- python-dotenv
- uvicorn（用于运行FastAPI服务器）

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/sydowma/codeReviewBot.git
   cd codeReviewBot
   ```

2. 安装所需包：
   ```
   pip install -r requirements.txt
   ```

3. 设置配置：
   - 创建一个`config.json`文件，包含您的GitLab、GitHub和OpenAI API凭证。
   - 或者，在`.env`文件中设置环境变量。

## 使用方法

### 作为CLI工具

```
python main.py <合并请求或拉取请求的url> [--summary-only]
```

### 作为FastAPI服务器

1. 启动服务器：
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. 向`http://localhost:8000/review`发送POST请求，JSON体如下：
   ```json
   {
     "url": "https://gitlab.com/your-project/merge_requests/1",
     "summary_only": false
   }
   ```

## 配置

该工具使用以下配置选项：

- `GITLAB_URL`：您的GitLab实例URL
- `GITLAB_TOKEN`：您的GitLab个人访问令牌
- `GITHUB_TOKEN`：您的GitHub个人访问令牌
- `OPENAI_API_KEY`：您的OpenAI API密钥
- `OPENAI_HTTP_PROXY`：（可选）OpenAI API调用的HTTP代理

您可以在`config.json`中设置这些选项，或者作为环境变量设置。

## 自定义

您可以通过修改`detailed_prompt.txt`和`summary_prompt.txt`文件来自定义审查提示。