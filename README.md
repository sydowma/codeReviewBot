# GitLab Code Review Bot

## English Version

### Introduction
GitLab Code Review Bot is an automated tool designed to streamline the code review process for GitLab merge requests. By leveraging OpenAI's GPT models, this bot provides intelligent, context-aware code reviews, helping developers identify potential issues and improve code quality.

### Features
- Automated code review for GitLab merge requests
- Two modes of operation:
  1. Detailed review with line-specific comments
  2. Summary-only review for a high-level overview
- Customizable review prompts
- Support for GitLab API and OpenAI API
- Command-line interface and FastAPI web server

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/sydowma/codeReviewBot.git
   cd codeReviewBot
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `GITLAB_URL`: Your GitLab instance URL
   - `GITLAB_TOKEN`: Your GitLab personal access token
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_HTTP_PROXY` (optional): HTTP proxy for OpenAI API calls

### Usage
1. Command-line interface:
   - For detailed review:
     ```
     python main.py <merge_request_url>
     ```
   - For summary-only review:
     ```
     python main.py <merge_request_url> --summary-only
     ```

2. FastAPI web server:
   - Start the server:
     ```
     python main.py
     ```
   - Send a POST request to `http://localhost:8000/review` with the following JSON body:
     ```json
     {
       "merge_request_url": "https://your-gitlab-instance.com/project/merge_requests/123",
       "summary_only": false
     }
     ```

### Customization
You can customize the review prompts by modifying the `detailed_prompt.txt` and `summary_prompt.txt` files in the project directory.

## 中文版本

### 简介
GitLab 代码审查机器人是一个自动化工具，旨在简化 GitLab 合并请求的代码审查过程。通过利用 OpenAI 的 GPT 模型，该机器人提供智能、上下文感知的代码审查，帮助开发者识别潜在问题并提高代码质量。

### 功能特性
- 自动化 GitLab 合并请求代码审查
- 两种运行模式：
  1. 详细审查，提供针对具体代码行的评论
  2. 仅摘要审查，提供高层次概述
- 可自定义审查提示词
- 支持 GitLab API 和 OpenAI API
- 命令行界面和 FastAPI 网络服务器

### 安装
1. 克隆仓库：
   ```
   git clone https://github.com/sydowma/codeReviewBot.git
   cd codeReviewBot
   ```
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
3. 设置环境变量：
   - `GITLAB_URL`：您的 GitLab 实例 URL
   - `GITLAB_TOKEN`：您的 GitLab 个人访问令牌
   - `OPENAI_API_KEY`：您的 OpenAI API 密钥
   - `OPENAI_HTTP_PROXY`（可选）：OpenAI API 调用的 HTTP 代理

### 使用方法
1. 命令行界面：
   - 详细审查：
     ```
     python main.py <合并请求URL>
     ```
   - 仅摘要审查：
     ```
     python main.py <合并请求URL> --summary-only
     ```

2. FastAPI 网络服务器：
   - 启动服务器：
     ```
     python main.py
     ```
   - 向 `http://localhost:8000/review` 发送 POST 请求，JSON 正文如下：
     ```json
     {
       "merge_request_url": "https://your-gitlab-instance.com/project/merge_requests/123",
       "summary_only": false
     }
     ```

### 自定义
您可以通过修改项目目录中的 `detailed_prompt.txt` 和 `summary_prompt.txt` 文件来自定义审查提示词。