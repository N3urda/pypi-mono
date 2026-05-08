"""Tests for core types."""

import pytest

from pypi_ai.types import (
    Api,
    Provider,
    ThinkingLevel,
    StopReason,
    TextContent,
    ThinkingContent,
    ImageContent,
    ToolCall,
    Usage,
    Cost,
    AssistantMessage,
    UserMessage,
    ToolResultMessage,
    Context,
    Model,
    Tool,
)


def test_api_enum():
    """Test Api enum values."""
    assert Api.ANTHROPIC_MESSAGES == "anthropic-messages"
    assert Api.OPENAI_COMPLETIONS == "openai-completions"
    assert Api.GOOGLE_GENERATIVE_AI == "google-generative-ai"


def test_provider_enum():
    """Test Provider enum values."""
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.GOOGLE == "google"


def test_text_content():
    """Test TextContent model."""
    content = TextContent(text="Hello world")

    assert content.type == "text"
    assert content.text == "Hello world"


def test_thinking_content():
    """Test ThinkingContent model."""
    content = ThinkingContent(thinking="Let me think...", redacted=False)

    assert content.type == "thinking"
    assert content.thinking == "Let me think..."


def test_image_content():
    """Test ImageContent model."""
    content = ImageContent(data="base64data", mime_type="image/png")

    assert content.type == "image"
    assert content.mime_type == "image/png"


def test_tool_call():
    """Test ToolCall model."""
    call = ToolCall(id="call_123", name="test_tool", arguments={"arg": "value"})

    assert call.type == "toolCall"
    assert call.id == "call_123"
    assert call.name == "test_tool"


def test_usage():
    """Test Usage model."""
    usage = Usage(input=100, output=50, cache_read=10)

    assert usage.input == 100
    assert usage.output == 50
    assert usage.cache_read == 10


def test_cost():
    """Test Cost model."""
    cost = Cost(input=0.001, output=0.002)

    assert cost.input == 0.001
    assert cost.output == 0.002


def test_assistant_message():
    """Test AssistantMessage model."""
    msg = AssistantMessage(
        content=[TextContent(text="Hello")],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="claude-test",
        stop_reason=StopReason.END,
    )

    assert msg.role == "assistant"
    assert len(msg.content) == 1


def test_user_message():
    """Test UserMessage model."""
    msg = UserMessage(content="Hello assistant")

    assert msg.role == "user"
    assert msg.content == "Hello assistant"


def test_user_message_with_content_list():
    """Test UserMessage with content list."""
    msg = UserMessage(content=[TextContent(text="Hello")])

    assert msg.role == "user"
    assert len(msg.content) == 1


def test_tool_result_message():
    """Test ToolResultMessage model."""
    msg = ToolResultMessage(
        tool_call_id="call_123",
        content=[TextContent(text="Result")],
        is_error=False,
    )

    assert msg.role == "toolResult"
    assert msg.tool_call_id == "call_123"


def test_context():
    """Test Context model."""
    ctx = Context(
        system_prompt="You are helpful",
        messages=[UserMessage(content="Hi")],
    )

    assert ctx.system_prompt == "You are helpful"
    assert len(ctx.messages) == 1


def test_model():
    """Test Model model."""
    model = Model(
        id="claude-test",
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
    )

    assert model.id == "claude-test"
    assert model.api == Api.ANTHROPIC_MESSAGES


def test_tool():
    """Test Tool model."""
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )

    assert tool.name == "test_tool"
    schema = tool.to_json_schema()
    assert schema["name"] == "test_tool"


def test_thinking_level_enum():
    """Test ThinkingLevel enum."""
    assert ThinkingLevel.MINIMAL == "minimal"
    assert ThinkingLevel.HIGH == "high"


def test_stop_reason_enum():
    """Test StopReason enum."""
    assert StopReason.END == "end"
    assert StopReason.TOOL_USE == "tool_use"
    assert StopReason.ERROR == "error"