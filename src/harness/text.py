"""Message content helpers.

With thinking enabled, Claude replies arrive as a *list* of content blocks
(thinking, text, tool_use...). Everything that stores or summarises a message
must reduce it to plain text first; this is the one place that does it.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def text_of(message: BaseMessage | str | list) -> str:
    content = message.content if isinstance(message, BaseMessage) else message
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
