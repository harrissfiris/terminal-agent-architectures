"""
Agent package — unified ReAct agent for TerminalBench tasks.

Quick start:
    from config import load_config
    from llm import create_llm
    from terminal.session import SafeTerminal
    from agent import ReActAgent, ToolExecutor

    config = load_config()
    llm = create_llm(config.llm)
    terminal = SafeTerminal(tmux_session)
    executor = ToolExecutor(terminal)
    agent = ReActAgent(llm, executor, config)
    result = agent.run("Solve the task...")
"""
from agent.react import ReActAgent, AgentResult
from agent.executor import ToolExecutor
from agent.memory import ConversationMemory
from agent.tools import get_tool_schemas, get_tool_names
from agent.prompts import get_system_prompt, build_initial_user_message
from agent.safety import CommandGuard, Verdict, SafetyResult

__all__ = [
    "ReActAgent",
    "AgentResult",
    "ToolExecutor",
    "ConversationMemory",
    "CommandGuard",
    "Verdict",
    "SafetyResult",
    "get_tool_schemas",
    "get_tool_names",
    "get_system_prompt",
    "build_initial_user_message",
]
