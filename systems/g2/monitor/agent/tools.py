"""
Tool definitions for the ReAct agent.

Each tool has:
  1. A Python execution function (takes SafeTerminal + args → str result)
  2. An OpenAI function-calling schema definition

Tools:
  - bash       : Execute a shell command
  - write_file : Write content to a file
  - patch_file : Surgical edit — replace a specific string in a file
  - read_file  : Read a file's contents
  - think      : Scratchpad for reasoning (no side effects)
  - submit     : Declare task complete with optional answer
"""
from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------
# OpenAI function-calling schemas
# ----------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command in the terminal. "
                "Returns the command output (stdout+stderr). "
                "Use for: running programs, installing packages, "
                "running tests, compiling, git operations, etc. "
                "Commands run in a persistent bash session — "
                "cd, env vars, etc. persist across calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Max seconds to wait (default 30). "
                            "Use higher values for slow commands "
                            "(pip install, make, training, etc.)."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file (creates or overwrites). "
                "Parent directories are created automatically. "
                "Use for: creating scripts, config files, patches, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path (e.g., /app/solve.py).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Surgically edit a file by replacing a specific text snippet. "
                "Much more efficient than write_file for small edits to large files. "
                "IMPORTANT: old_text must match EXACTLY (including whitespace/indentation). "
                "Use read_file first to see the exact text you want to replace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to edit.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": (
                            "The exact text to find and replace. "
                            "Must match precisely including whitespace. "
                            "Include 2-3 lines of context for uniqueness."
                        ),
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. "
                "Returns the file text (truncated if very large). "
                "Use for: inspecting source code, configs, data files, logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Private reasoning scratchpad — no side effects. "
                "Use to: plan next steps, analyze errors, reason about "
                "approach, break down complex problems, reflect on progress. "
                "Content is added to your context but not executed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Your reasoning, analysis, or plan.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Declare the task as completed and optionally provide a final answer. "
                "IMPORTANT: Only call this when you are confident the task is solved. "
                "If the task has tests, make sure they pass before submitting. "
                "If the task produces output files, verify they exist and look correct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Short final answer or confirmation message. "
                            "Keep it concise."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ----------------------------------------------------------------
# Enhanced tools — unlocked via THESIS_AGENT_TOOLS=enhanced
# ----------------------------------------------------------------

ENHANCED_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for a text pattern across files (like grep -rn). "
                "Returns matching lines with file paths and line numbers. "
                "Use for: finding function definitions, locating imports, "
                "searching for error strings, discovering usage patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory to search in (default: /app). "
                            "Use absolute path."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List the contents of a directory with file types and sizes. "
                "Returns a structured listing (like ls -la). "
                "Use for: exploring project structure, finding files, "
                "checking what exists before reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the directory to list "
                            "(default: /app)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


def get_tool_schemas(mode: str = "default") -> list[dict[str, Any]]:
    """Return tool schemas based on the active tool set.

    Args:
        mode: ``"default"`` for the 6 base tools, ``"enhanced"`` to also
              include ``search_files`` and ``list_dir``.
    """
    if mode == "enhanced":
        return TOOL_SCHEMAS + ENHANCED_TOOL_SCHEMAS
    return TOOL_SCHEMAS


def get_tool_names(mode: str = "default") -> list[str]:
    """Return list of available tool names."""
    return [t["function"]["name"] for t in get_tool_schemas(mode)]
