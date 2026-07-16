from __future__ import annotations as _annotations

from typing import Literal

import pytest

from pydantic_ai import Agent

from ..conftest import try_import
from .mock_openai import MockOpenAI, completion_message

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.nscale import NscaleProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


async def test_nscale_agent_with_mock(allow_model_requests: None) -> None:
    """Nscale models work through the normal public Agent API."""
    mock_client = MockOpenAI.create_mock(
        completion_message(ChatCompletionMessage.model_construct(content='Nscale tests are working.', role='assistant'))
    )
    model = OpenAIChatModel('meta-llama/Llama-3.1-8B-Instruct', provider=NscaleProvider(openai_client=mock_client))

    result = await Agent(model).run('Are the Nscale tests working?')

    assert result.output == 'Nscale tests are working.'


def _text_chunk(
    text: str,
    *,
    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None = None,
) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id='test-response',
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=text, role='assistant'),
                finish_reason=finish_reason,
            )
        ],
        created=0,
        model='meta-llama/Llama-3.1-8B-Instruct',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=2, prompt_tokens=4, total_tokens=6),
    )


async def test_nscale_agent_stream_with_mock(allow_model_requests: None) -> None:
    """Nscale models stream ordinary text through the normal public Agent API."""
    mock_client = MockOpenAI.create_mock_stream(
        [_text_chunk('Nscale streaming '), _text_chunk('works.', finish_reason='stop')]
    )
    model = OpenAIChatModel('meta-llama/Llama-3.1-8B-Instruct', provider=NscaleProvider(openai_client=mock_client))

    async with Agent(model).run_stream('Does Nscale streaming work?') as result:
        assert await result.get_output() == 'Nscale streaming works.'


@pytest.mark.vcr
async def test_nscale_agent(allow_model_requests: None, nscale_service_token: str) -> None:
    """A standard Nscale model works through the public Agent API."""
    model = OpenAIChatModel(
        'meta-llama/Llama-3.1-8B-Instruct', provider=NscaleProvider(service_token=nscale_service_token)
    )

    result = await Agent(model).run('What is the capital of France? Answer in one sentence.')

    assert result.output == 'The capital of France is Paris.'


@pytest.mark.vcr
async def test_nscale_agent_stream(allow_model_requests: None, nscale_service_token: str) -> None:
    """A standard Nscale model streams text through the public Agent API."""
    model = OpenAIChatModel(
        'meta-llama/Llama-3.1-8B-Instruct', provider=NscaleProvider(service_token=nscale_service_token)
    )

    async with Agent(model).run_stream('What is the capital of France? Answer in one sentence.') as result:
        assert await result.get_output() == 'The capital of France is Paris.'
