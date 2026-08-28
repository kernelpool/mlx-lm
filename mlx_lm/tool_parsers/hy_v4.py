# Copyright © 2026 Apple Inc.

"""
Tool parser for Tencent Hy4 (HYV4).

Reference:
https://github.com/vllm-project/vllm/pull/54160

Format:
    <tool_calls:opensource>
    <tool_call:opensource>function_name
    <arg_key:opensource>key1</arg_key:opensource>
    <arg_value:opensource>value1</arg_value:opensource>
    ...
    </tool_call:opensource>
    ...
    </tool_calls:opensource>

The regexes below also accept the unsuffixed token names.
"""

import ast
import json
from typing import Any

import regex as re

tool_call_start = "<tool_calls:opensource>"
tool_call_end = "</tool_calls:opensource>"

_SUFFIX = r"(?::[\w-]+)?"

_tool_call_regex = re.compile(
    rf"<tool_call{_SUFFIX}>(.*?)</tool_call{_SUFFIX}>",
    re.DOTALL,
)
_arg_pair_regex = re.compile(
    rf"<arg_key{_SUFFIX}>(.*?)</arg_key{_SUFFIX}>"
    rf"(?:\\n|\s)*"
    rf"<arg_value{_SUFFIX}>(.*?)</arg_value{_SUFFIX}>",
    re.DOTALL,
)
_arg_key_regex = re.compile(rf"<arg_key{_SUFFIX}>")
_tool_call_open_regex = re.compile(rf"\s*<tool_call{_SUFFIX}>")


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        arg_type = params.get("properties", {}).get(arg_name, {}).get("type")
        return arg_type == "string"
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _parse_single_call(text: str, tools: list[Any] | None):
    func_name = _arg_key_regex.split(text, 1)[0].strip()

    arg_dct: dict[str, Any] = {}
    for key, value in _arg_pair_regex.findall(text):
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(func_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val
    return dict(name=func_name, arguments=arg_dct)


def parse_tool_call(text: str, tools: list[Any] | None = None):
    matches = _tool_call_regex.findall(text)
    if matches:
        calls = [_parse_single_call(body, tools) for body in matches]
        return calls[0] if len(calls) == 1 else calls

    # Truncated call without the closing </tool_call> tag
    open_match = _tool_call_open_regex.match(text)
    if open_match:
        text = text[open_match.end() :]
    return _parse_single_call(text.strip(), tools)
