from typing import List, Dict
import tiktoken


class ReviewRequestBuilder:
    def __init__(self):
        self.summary_prompt = ""
        self.detailed_prompt = ""
        self.max_tokens = 26500
        self.encoding = tiktoken.get_encoding("cl100k_base")  # 使用 OpenAI 的 tokenizer

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.encoding.encode(text))

    def truncate_diff(self, diff: str, max_tokens: int) -> str:
        """智能截断 diff 内容，保留最重要的部分"""
        tokens = self.encoding.encode(diff)
        if len(tokens) <= max_tokens:
            return diff

        # 保留文件开头和结尾的变更内容
        start_tokens = tokens[:max_tokens // 2]
        end_tokens = tokens[-max_tokens // 2:]

        truncated_diff = (
                self.encoding.decode(start_tokens) +
                "\n...[部分内容已省略]...\n" +
                self.encoding.decode(end_tokens)
        )
        return truncated_diff

    def filter_changes(self, changes: List[Dict], available_tokens: int) -> List[Dict]:
        """根据重要性过滤和裁剪变更内容"""
        filtered_changes = []
        tokens_used = 0

        # 按文件重要性排序（可以根据实际需求调整排序逻辑）
        sorted_changes = sorted(changes, key=lambda x: self._get_change_priority(x))

        for change in sorted_changes:
            diff = change['diff']
            path = change['new_path']

            # 计算当前文件需要的 token
            file_tokens = self.count_tokens(f"File: {path}\n\n{diff}")

            if tokens_used + file_tokens > available_tokens:
                # 如果整个文件太大，尝试截断
                remaining_tokens = available_tokens - tokens_used
                if remaining_tokens > 500:  # 确保至少有足够的令牌来显示有意义的内容
                    truncated_diff = self.truncate_diff(diff, remaining_tokens - 100)  # 预留一些token给文件路径
                    filtered_changes.append({
                        'new_path': path,
                        'diff': truncated_diff
                    })
                    tokens_used += self.count_tokens(f"File: {path}\n\n{truncated_diff}")
                break

            filtered_changes.append(change)
            tokens_used += file_tokens

        return filtered_changes

    def _get_change_priority(self, change: Dict) -> int:
        """确定文件变更的优先级，返回优先级分数（越小越重要）"""
        path = change['new_path'].lower()

        # 优先级规则
        if any(key in path for key in ['readme', 'config', 'main', 'core']):
            return 1
        elif path.endswith(('.py', '.java', '.cpp', '.js')):
            return 2
        elif path.endswith(('test.py', 'test.java', 'test.js')):
            return 3
        elif path.endswith(('.md', '.txt')):
            return 4
        elif path.endswith(('.css', '.html')):
            return 5
        return 6

    def build_review_request(self, changes: Dict, summary_only: bool, mr_information: str, custom_prompt: str) -> str:
        """构建审查请求，确保不超过 token 限制"""
        # 计算基础内容的 token 数量
        base_prompt = self.summary_prompt if summary_only else self.detailed_prompt
        if len(custom_prompt) > 0:
            fixed_content = f"{custom_prompt}"
        else:
            fixed_content = f"{base_prompt}\n{mr_information}"
        fixed_tokens = self.count_tokens(fixed_content)

        # 计算可用于文件内容的 token 数量
        available_tokens = self.max_tokens - fixed_tokens - 100  # 预留一些 token 作为缓冲

        if available_tokens <= 0:
            raise ValueError("基础提示内容已超过 token 限制")

        # 过滤和裁剪变更内容
        filtered_changes = self.filter_changes(changes['changes'], available_tokens)

        # 构建文件内容
        files_content = []
        for change in filtered_changes:
            files_content.append(f"File: {change['new_path']}\n\n{change['diff']}")

        # 组合最终的请求内容
        final_content = fixed_content + "\n" + "\n".join(files_content)

        # 最终验证
        if self.count_tokens(final_content) > self.max_tokens:
            raise ValueError("Despite filtering, content still exceeds token limit")

        return final_content