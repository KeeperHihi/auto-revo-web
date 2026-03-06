#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = APP_ROOT / "config.json"
DEFAULT_SYSTEM_PROMPT_FILE = "prompts/sys-prompt.md"
DEFAULT_USER_PROMPT_FILE = "prompts/user-prompt.md"
SESSION_ID_PATTERN = re.compile(r"session id:\s*([0-9a-f-]{36})", re.IGNORECASE)


@dataclass
class LlmAccessConfig:
    url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class CodexConfig:
    command: str = "codex"
    model: str = "gpt-5.3-codex-xhigh"
    profile: str = ""
    dangerous_bypass: bool = True
    timeout_seconds: int = 1800
    retries: int = 3
    extra_args: list[str] = field(
        default_factory=lambda: ["-c", 'model_reasoning_effort="xhigh"']
    )
    dry_run: bool = False
    auto_git_commit: bool = True
    auto_git_push: bool = True
    git_remote: str = "origin"
    git_branch: str = "main"
    git_commit_prefix: str = ""


@dataclass
class AppConfig:
    site_name: str = "demo"
    iterations: int = 3
    interval_seconds: int = 30
    append_iteration_context: bool = True
    system_prompt_file: str = DEFAULT_SYSTEM_PROMPT_FILE
    user_prompt_file: str = DEFAULT_USER_PROMPT_FILE
    llm_access: LlmAccessConfig = field(default_factory=LlmAccessConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)


def strip_json_comments(content: str) -> str:
    result: list[str] = []
    in_string = False
    string_char = ""
    in_single_comment = False
    in_multi_comment = False

    i = 0
    while i < len(content):
        char = content[i]
        next_char = content[i + 1] if i + 1 < len(content) else ""

        if in_single_comment:
            if char == "\n":
                in_single_comment = False
                result.append(char)
            i += 1
            continue

        if in_multi_comment:
            if char == "*" and next_char == "/":
                in_multi_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            result.append(char)
            if char == "\\" and next_char:
                result.append(next_char)
                i += 2
                continue
            if char == string_char:
                in_string = False
                string_char = ""
            i += 1
            continue

        if char in ('"', "'"):
            in_string = True
            string_char = char
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            in_single_comment = True
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_multi_comment = True
            i += 2
            continue

        result.append(char)
        i += 1

    return "".join(result)


def to_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, number)


def to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


def to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def to_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def normalize_branch_name(branch_name: str) -> str:
    return str(branch_name or "").strip().replace("refs/heads/", "")


def normalize_config(raw: dict[str, Any]) -> AppConfig:
    config = AppConfig()

    llm_access_raw = raw.get("llmAccess", {}) if isinstance(raw.get("llmAccess"), dict) else {}
    codex_raw = raw.get("codex", {}) if isinstance(raw.get("codex"), dict) else {}
    evolution_raw = raw.get("evolution", {}) if isinstance(raw.get("evolution"), dict) else {}

    if "siteName" in raw:
        config.site_name = to_str(raw.get("siteName"), config.site_name)
    if "iterations" in raw:
        config.iterations = to_int(raw.get("iterations"), config.iterations, minimum=1)
    if "iteration" in raw:
        config.iterations = to_int(raw.get("iteration"), config.iterations, minimum=1)
    if "intervalSeconds" in raw:
        config.interval_seconds = to_int(raw.get("intervalSeconds"), config.interval_seconds, minimum=0)
    if "appendIterationContext" in raw:
        config.append_iteration_context = to_bool(
            raw.get("appendIterationContext"), config.append_iteration_context
        )
    if "systemPromptFile" in raw:
        config.system_prompt_file = to_str(raw.get("systemPromptFile"), config.system_prompt_file)
    if "userPromptFile" in raw:
        config.user_prompt_file = to_str(raw.get("userPromptFile"), config.user_prompt_file)

    if "defaultIterations" in evolution_raw and "iterations" not in raw:
        config.iterations = to_int(evolution_raw.get("defaultIterations"), config.iterations, minimum=1)
    if "intervalMs" in evolution_raw and "intervalSeconds" not in raw:
        config.interval_seconds = to_int(evolution_raw.get("intervalMs"), config.interval_seconds * 1000, minimum=0) // 1000
    if "appendIterationContext" in evolution_raw and "appendIterationContext" not in raw:
        config.append_iteration_context = to_bool(
            evolution_raw.get("appendIterationContext"), config.append_iteration_context
        )
    if "systemPromptFile" in evolution_raw and "systemPromptFile" not in raw:
        config.system_prompt_file = to_str(
            evolution_raw.get("systemPromptFile"), config.system_prompt_file
        )

    if llm_access_raw:
        config.llm_access.url = to_str(llm_access_raw.get("url"), config.llm_access.url)
        config.llm_access.api_key = to_str(llm_access_raw.get("apiKey"), config.llm_access.api_key)
        config.llm_access.model = to_str(llm_access_raw.get("model"), config.llm_access.model)

    if codex_raw:
        if "gitBranch" in codex_raw and "siteName" not in raw:
            config.site_name = to_str(codex_raw.get("gitBranch"), config.site_name)
        if "command" in codex_raw:
            config.codex.command = to_str(codex_raw.get("command"), config.codex.command)
        if "model" in codex_raw:
            config.codex.model = to_str(codex_raw.get("model"), config.codex.model)
        if "profile" in codex_raw:
            config.codex.profile = to_str(codex_raw.get("profile"), config.codex.profile)
        if "dangerouslyBypassApprovalsAndSandbox" in codex_raw:
            config.codex.dangerous_bypass = to_bool(
                codex_raw.get("dangerouslyBypassApprovalsAndSandbox"),
                config.codex.dangerous_bypass,
            )
        if "timeoutSeconds" in codex_raw:
            config.codex.timeout_seconds = to_int(
                codex_raw.get("timeoutSeconds"), config.codex.timeout_seconds, minimum=1
            )
        if "timeoutMs" in codex_raw and "timeoutSeconds" not in codex_raw:
            config.codex.timeout_seconds = to_int(
                codex_raw.get("timeoutMs"), config.codex.timeout_seconds * 1000, minimum=1000
            ) // 1000
        if "retries" in codex_raw:
            config.codex.retries = to_int(codex_raw.get("retries"), config.codex.retries, minimum=0)
        if "reconnectingRounds" in codex_raw and "retries" not in codex_raw:
            config.codex.retries = to_int(
                codex_raw.get("reconnectingRounds"), config.codex.retries, minimum=0
            )
        if "extraArgs" in codex_raw:
            extra_args = to_str_list(codex_raw.get("extraArgs"))
            if extra_args:
                config.codex.extra_args = extra_args
        if "dryRun" in codex_raw:
            config.codex.dry_run = to_bool(codex_raw.get("dryRun"), config.codex.dry_run)
        if "autoGitCommit" in codex_raw:
            config.codex.auto_git_commit = to_bool(codex_raw.get("autoGitCommit"), config.codex.auto_git_commit)
        if "autoGitPush" in codex_raw:
            config.codex.auto_git_push = to_bool(codex_raw.get("autoGitPush"), config.codex.auto_git_push)
        if "gitRemote" in codex_raw:
            config.codex.git_remote = to_str(codex_raw.get("gitRemote"), config.codex.git_remote)
        if "gitBranch" in codex_raw:
            config.codex.git_branch = to_str(codex_raw.get("gitBranch"), config.codex.git_branch)
        if "gitCommitPrefix" in codex_raw:
            config.codex.git_commit_prefix = to_str(
                codex_raw.get("gitCommitPrefix"), config.codex.git_commit_prefix
            )

    config.site_name = config.site_name.strip()
    config.iterations = max(1, config.iterations)
    config.interval_seconds = max(0, config.interval_seconds)
    config.codex.timeout_seconds = max(1, config.codex.timeout_seconds)
    config.codex.retries = max(0, config.codex.retries)
    config.codex.git_remote = config.codex.git_remote.strip() or "origin"
    config.codex.git_branch = normalize_branch_name(config.codex.git_branch) or "main"
    if not config.site_name:
        raise ValueError("siteName 不能为空")
    return config


def load_config(config_file: Path) -> AppConfig:
    if not config_file.exists():
        raise FileNotFoundError(
            f"未找到配置文件: {config_file}，请先从 config.template.json 复制生成 config.json"
        )

    content = config_file.read_text(encoding="utf-8")
    try:
        parsed = json.loads(strip_json_comments(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"config.json 格式错误: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("config.json 根节点必须是对象")
    return normalize_config(parsed)


def resolve_local_path_from_root(path_value: str, field_name: str) -> Path:
    if not path_value:
        raise ValueError(f"{field_name} 不能为空")

    candidate = Path(path_value)
    absolute = candidate.resolve() if candidate.is_absolute() else (APP_ROOT / candidate).resolve()
    root = APP_ROOT.resolve()
    if absolute != root and root not in absolute.parents:
        raise ValueError(f"{field_name} 必须位于项目根目录内部")
    return absolute


def read_text_file(path: Path, field_name: str, allow_empty: bool = False) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"读取 {field_name} 失败: {exc}") from exc

    text = content.strip()
    if not allow_empty and not text:
        raise ValueError(f"{field_name} 为空: {path}")
    return text


def build_llm_runtime_hint(config: LlmAccessConfig) -> str:
    if not (config.url and config.api_key and config.model):
        return ""

    return "\n".join(
        [
            "- 可选外部模型调用（运行时注入）：",
            f"  - url: {config.url}",
            f"  - model: {config.model}",
            "  - api_key_env: LLM_ACCESS_API_KEY（只读环境变量，禁止输出明文）",
        ]
    )


def render_system_prompt(template: str, llm_config: LlmAccessConfig) -> str:
    runtime_hint = build_llm_runtime_hint(llm_config)
    token = "{{LLM_RUNTIME_HINT}}"
    rendered = template
    if token in rendered:
        rendered = rendered.replace(token, runtime_hint)
    elif runtime_hint:
        rendered = f"{rendered.strip()}\n\n{runtime_hint}".strip()
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


def resolve_workspace(site_name: str) -> Path:
    webs_root = (APP_ROOT / "webs").resolve()
    webs_root.mkdir(parents=True, exist_ok=True)
    workspace = (webs_root / site_name).resolve()

    if workspace != webs_root and webs_root not in workspace.parents:
        raise ValueError("siteName 非法，必须位于 webs 目录内")

    if not workspace.exists() or not workspace.is_dir():
        raise FileNotFoundError(
            f"未找到站点目录: {workspace}\n"
            "请先创建空仓库目录，例如: mkdir -p webs/<siteName> && cd webs/<siteName> && git init"
        )

    return workspace


def run_git(workspace: Path, args: list[str], timeout_seconds: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 git 命令，请先安装 git") from exc


def ensure_workspace_is_git_repo(workspace: Path) -> None:
    check = run_git(workspace, ["rev-parse", "--is-inside-work-tree"])
    if check.returncode != 0 or check.stdout.strip() != "true":
        raise RuntimeError(
            f"{workspace} 不是 git 仓库。\n"
            "请先在该目录执行 git init，并按需配置远端。"
        )


def get_current_branch_name(workspace: Path) -> str:
    result = run_git(workspace, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"读取当前分支失败: {details}")
    return normalize_branch_name(result.stdout)


def ensure_branch_ready(workspace: Path, branch_name: str) -> None:
    current = get_current_branch_name(workspace)
    target = normalize_branch_name(branch_name)
    if current == target:
        return

    exists = run_git(workspace, ["show-ref", "--verify", "--quiet", f"refs/heads/{target}"])
    if exists.returncode == 0:
        switch = run_git(workspace, ["checkout", target])
    else:
        switch = run_git(workspace, ["checkout", "-B", target])

    if switch.returncode != 0:
        details = (switch.stderr or switch.stdout).strip()
        raise RuntimeError(f"切换到分支 {target} 失败: {details}")

    print(f"[GIT] 已切换到分支: {target}")


def ensure_remote_ready(workspace: Path, remote_name: str) -> None:
    result = run_git(workspace, ["remote", "get-url", remote_name])
    if result.returncode != 0:
        raise RuntimeError(
            f"未找到远端 {remote_name}。\n"
            f"请先在 {workspace} 执行: git remote add {remote_name} <你的仓库地址>"
        )


def inspect_workspace_state(workspace: Path) -> str:
    has_commit = run_git(workspace, ["rev-parse", "--verify", "HEAD"]).returncode == 0
    tracked = run_git(workspace, ["ls-files"])
    if tracked.returncode != 0:
        raise RuntimeError(f"读取仓库文件列表失败：{extract_tail(tracked.stderr or tracked.stdout, 400)}")
    tracked_files = [line.strip() for line in tracked.stdout.splitlines() if line.strip()]

    uncommitted = run_git(workspace, ["status", "--porcelain"])
    if uncommitted.returncode != 0:
        raise RuntimeError(f"读取仓库状态失败：{extract_tail(uncommitted.stderr or uncommitted.stdout, 400)}")
    pending = [line.strip() for line in uncommitted.stdout.splitlines() if line.strip()]

    if not has_commit and not tracked_files and not pending:
        return "empty"
    return "non_empty"


def extract_tail(text: str, max_length: int = 1200) -> str:
    if len(text) <= max_length:
        return text.strip()
    return f"...{text[-max_length:].strip()}"


def extract_session_id(text: str) -> str:
    match = SESSION_ID_PATTERN.search(text or "")
    return match.group(1) if match else ""


def sanitize_commit_message(message: str) -> str:
    return " ".join(str(message or "").split()).strip()[:180]


def extract_codex_commit_message(output: str) -> str:
    patterns = [
        re.compile(r"^\s*COMMIT_MESSAGE\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*提交信息\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*commit\s+message\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    ]

    for pattern in patterns:
        match = pattern.search(output or "")
        if not match:
            continue
        normalized = sanitize_commit_message(match.group(1))
        if normalized:
            return normalized

    return ""


def build_iteration_prompt(
    system_prompt: str,
    user_prompt: str,
    iteration: int,
    total_iterations: int,
    previous_tail: str,
    append_iteration_context: bool,
) -> str:
    sections: list[str] = [
        "【系统提示词】",
        system_prompt.strip(),
        "",
        "【用户创意】",
        user_prompt.strip(),
        "",
    ]

    if append_iteration_context:
        sections.extend(
            [
                "【本轮迭代上下文】",
                f"- 轮次：第 {iteration}/{total_iterations} 轮",
                f"- 时间：{datetime.now(timezone.utc).isoformat()}",
            ]
        )
        if previous_tail:
            sections.extend(["- 上轮输出摘要（截断）：", previous_tail])
        sections.extend(
            [
                "- 要求：基于当前仓库最新状态继续推进，不要重复上一轮内容。",
                "",
            ]
        )

    sections.extend(
        [
            "【执行要求】",
            "1. 先审查当前仓库状态，选出本轮最有价值且可交付的改进。",
            "2. 直接修改代码并确保项目可运行。",
            "3. 至少执行一条有效验证命令（例如构建、测试或语法检查）。",
            "4. 结尾说明：本轮改动、验证结果、下一轮建议。",
            "5. 若本轮有代码变更，请最后单独输出：COMMIT_MESSAGE: <提交信息>。",
        ]
    )

    return "\n".join(sections).strip()


def build_codex_args(config: AppConfig, workspace: Path, resume_session_id: str) -> list[str]:
    args = (
        ["exec", "resume", resume_session_id]
        if resume_session_id
        else ["exec", "--cd", str(workspace), "--color", "never"]
    )

    if config.codex.model:
        args.extend(["--model", config.codex.model])
    if config.codex.profile:
        args.extend(["--profile", config.codex.profile])
    if config.codex.dangerous_bypass:
        args.append("--dangerously-bypass-approvals-and-sandbox")

    args.extend(config.codex.extra_args)
    args.append("-")
    return args


def build_codex_env(config: AppConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.llm_access.api_key:
        env["LLM_ACCESS_API_KEY"] = config.llm_access.api_key
    return env


def run_codex_iteration(
    config: AppConfig,
    workspace: Path,
    prompt: str,
    incoming_session_id: str,
) -> tuple[str, str, str]:
    command = config.codex.command
    timeout_seconds = config.codex.timeout_seconds
    retries = config.codex.retries
    session_id = incoming_session_id
    env = build_codex_env(config)

    for attempt in range(retries + 1):
        args = build_codex_args(config, workspace, session_id)
        rendered_cmd = " ".join([shlex.quote(command), *[shlex.quote(item) for item in args]])
        print(f"[系统] 启动命令（{attempt + 1}/{retries + 1}）：{rendered_cmd}")

        try:
            result = subprocess.run(
                [command, *args],
                cwd=str(workspace),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"未找到 Codex 命令：{command}。请先确认 Codex CLI 可用。"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            combined = f"{exc.stdout or ''}\n{exc.stderr or ''}"
            observed = extract_session_id(combined)
            if observed:
                session_id = observed

            if attempt >= retries:
                raise RuntimeError(f"Codex 执行超时（{timeout_seconds} 秒），且重试次数已耗尽") from exc

            print(f"[警告] Codex 超时，2 秒后重试（{attempt + 1}/{retries}）")
            time.sleep(2)
            continue

        combined = f"{result.stdout}\n{result.stderr}"
        observed = extract_session_id(combined)
        if observed:
            session_id = observed

        if result.stdout.strip():
            print("[Codex标准输出]")
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("[Codex标准错误]", file=sys.stderr)
            print(result.stderr.rstrip(), file=sys.stderr)

        commit_message = extract_codex_commit_message(combined)
        if result.returncode == 0:
            return session_id, extract_tail(combined), commit_message

        if attempt >= retries:
            raise RuntimeError(
                f"Codex 执行失败（退出码 {result.returncode}）：{extract_tail(combined, 1500)}"
            )

        print(f"[警告] Codex 执行失败，2 秒后重试（{attempt + 1}/{retries}）")
        time.sleep(2)

    raise RuntimeError("重试循环异常结束")


def count_changed_files(workspace: Path) -> int:
    status = run_git(workspace, ["status", "--porcelain"])
    if status.returncode != 0:
        return -1
    return len([line for line in status.stdout.splitlines() if line.strip()])


def build_commit_message(config: AppConfig, codex_message: str, iteration: int) -> str:
    base = sanitize_commit_message(codex_message) or f"第{iteration}轮自动进化更新"
    prefix = sanitize_commit_message(config.codex.git_commit_prefix)
    if prefix and not base.startswith(prefix):
        return f"{prefix} {base}".strip()
    return base


def commit_and_push_changes(
    config: AppConfig,
    workspace: Path,
    codex_message: str,
    iteration: int,
) -> tuple[bool, bool]:
    if not config.codex.auto_git_commit:
        print("[GIT] 已关闭自动提交（autoGitCommit=false）")
        return False, False

    add_result = run_git(workspace, ["add", "-A"], timeout_seconds=120)
    if add_result.returncode != 0:
        raise RuntimeError(f"git add 失败：{extract_tail(add_result.stderr or add_result.stdout, 600)}")

    staged = run_git(workspace, ["diff", "--cached", "--name-only"])
    if staged.returncode != 0:
        raise RuntimeError(f"读取暂存区失败：{extract_tail(staged.stderr or staged.stdout, 600)}")

    staged_files = [line.strip() for line in staged.stdout.splitlines() if line.strip()]
    if not staged_files:
        print("[GIT] 未检测到可提交改动，跳过提交与推送")
        return False, False

    message = build_commit_message(config, codex_message, iteration)
    commit_result = run_git(workspace, ["commit", "-m", message], timeout_seconds=120)
    if commit_result.returncode != 0:
        details = extract_tail(commit_result.stderr or commit_result.stdout, 1000)
        if "Author identity unknown" in details:
            raise RuntimeError(
                "git commit 失败：未配置用户信息，请先执行\n"
                "git config user.name \"<你的名字>\"\n"
                "git config user.email \"<你的邮箱>\""
            )
        raise RuntimeError(f"git commit 失败：{details}")

    print(f"[GIT] 已提交：{message}")

    if not config.codex.auto_git_push:
        print("[GIT] 已关闭自动推送（autoGitPush=false）")
        return True, False

    remote = config.codex.git_remote
    branch = normalize_branch_name(config.codex.git_branch)
    push_result = run_git(workspace, ["push", "-u", remote, branch], timeout_seconds=180)
    if push_result.returncode != 0:
        raise RuntimeError(f"git push 失败：{extract_tail(push_result.stderr or push_result.stdout, 1000)}")

    print(f"[GIT] 已推送到 {remote}/{branch}")
    return True, True


def ask_user_prompt() -> str:
    try:
        return input("请输入你的一句话网站创意：").strip()
    except EOFError:
        return ""


def resolve_user_prompt(cli_prompt: str | None, config: AppConfig) -> str:
    if cli_prompt and cli_prompt.strip():
        return cli_prompt.strip()

    prompt_file = resolve_local_path_from_root(config.user_prompt_file, "userPromptFile")
    file_prompt = read_text_file(prompt_file, "userPromptFile", allow_empty=True)
    if file_prompt:
        print(f"[系统] 已从文件读取用户创意：{prompt_file}")
        return file_prompt

    if sys.stdin.isatty():
        interactive_prompt = ask_user_prompt()
        if interactive_prompt:
            return interactive_prompt

    raise ValueError(
        f"用户创意为空，请填写 {prompt_file}，或通过 --prompt 参数传入一句创意"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动进化脚本入口（在 webs/<siteName> 独立仓库中执行迭代）",
        usage="python3 evolution.py [--site 站点名] [--iteration 轮次] [--prompt 创意] [--dry-run]",
        add_help=False,
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--site", help="覆盖 config.json 中的 siteName（目标站点目录名）")
    parser.add_argument(
        "--iterations",
        "--iteration",
        dest="iterations",
        type=int,
        help="覆盖 config.json 中的迭代轮次",
    )
    parser.add_argument("--prompt", help="直接传入一句网站创意，优先级高于 userPromptFile")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅演练流程与提示词组装，不调用 Codex，也不执行 git 提交/推送",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    config = load_config(CONFIG_FILE)

    if args.site:
        config.site_name = args.site.strip()
    if args.iterations is not None:
        config.iterations = max(1, int(args.iterations))

    workspace = resolve_workspace(config.site_name)
    ensure_workspace_is_git_repo(workspace)

    system_prompt_path = resolve_local_path_from_root(config.system_prompt_file, "systemPromptFile")
    system_prompt_template = read_text_file(system_prompt_path, "systemPromptFile")
    system_prompt = render_system_prompt(system_prompt_template, config.llm_access)

    user_prompt = resolve_user_prompt(args.prompt, config)

    dry_run = args.dry_run or config.codex.dry_run
    total_iterations = config.iterations

    print(f"[系统] 目标站点目录：{workspace}")
    print(f"[系统] 迭代轮次：{total_iterations}")
    print(f"[系统] 演练模式：{dry_run}")

    if config.codex.auto_git_push and not config.codex.auto_git_commit:
        raise ValueError("配置错误：autoGitPush=true 时必须同时设置 autoGitCommit=true")

    target_branch = normalize_branch_name(config.codex.git_branch)
    ensure_branch_ready(workspace, target_branch)
    if not dry_run and config.codex.auto_git_push:
        ensure_remote_ready(workspace, config.codex.git_remote)

    workspace_state = inspect_workspace_state(workspace)
    if workspace_state == "empty":
        print("[系统] 检测到空仓库，将从 0 开始生成网站")
    else:
        print("[系统] 检测到仓库已有内容，将在现有基础上继续进化")

    previous_tail = ""
    resume_session_id = ""

    for iteration in range(1, total_iterations + 1):
        print(f"[自动进化] 第 {iteration}/{total_iterations} 轮开始")
        prompt = build_iteration_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            iteration=iteration,
            total_iterations=total_iterations,
            previous_tail=previous_tail,
            append_iteration_context=config.append_iteration_context,
        )

        codex_commit_message = ""
        if dry_run:
            preview = extract_tail(prompt, 800)
            previous_tail = preview
            print("[演练] 本轮提示词摘要：")
            print(preview)
        else:
            resume_session_id, previous_tail, codex_commit_message = run_codex_iteration(
                config=config,
                workspace=workspace,
                prompt=prompt,
                incoming_session_id=resume_session_id,
            )
            commit_and_push_changes(
                config=config,
                workspace=workspace,
                codex_message=codex_commit_message,
                iteration=iteration,
            )

        changed_count = count_changed_files(workspace)
        if changed_count >= 0:
            print(f"[自动进化] 第 {iteration} 轮完成，当前仓库未提交文件数：{changed_count}")
        else:
            print(f"[自动进化] 第 {iteration} 轮完成")

        if iteration < total_iterations and config.interval_seconds > 0:
            print(f"[自动进化] 等待 {config.interval_seconds} 秒后进入下一轮")
            time.sleep(config.interval_seconds)

    print(f"[系统] 进化结束，共执行 {total_iterations} 轮")
    return 0


def main() -> None:
    try:
        sys.exit(run())
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
