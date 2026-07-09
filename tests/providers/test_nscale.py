
import httpx
import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import TestEnv, try_import
from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai import OpenAIError
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.nscale import NscaleProvider


def test_nscale_provider():
    provider = NscaleProvider(service_token='foobar')
    assert provider.name == 'nscale'
    assert provider.base_url == 'https://inference.api.nscale.com/v1'
    assert provider.client.api_key == 'foobar'


def test_nscale_provider_custom_base_url():
    provider = NscaleProvider(service_token='foobar', base_url='https://example.com/v1')
    assert provider.base_url == 'https://example.com/v1'
    assert str(provider.client.base_url) == 'https://example.com/v1/'


def test_nscale_provider_uses_env_service_token(env: TestEnv):
    env.set('NSCALE_SERVICE_TOKEN', 'env-token')
    provider = NscaleProvider()
    assert provider.client.api_key == 'env-token'


def test_nscale_glm_model_profile():
    profile = NscaleProvider.model_profile('glm/glm-5.2-fp8')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True
    assert profile.openai_chat_thinking_field == 'reasoning_content'
    assert profile.openai_supports_tool_choice_required is True


def test_nscale_glm_model_inference(env: TestEnv):
    env.set('NSCALE_SERVICE_TOKEN', 'env-token')
    model = infer_model('nscale:glm/glm-5.2-fp8')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'glm/glm-5.2-fp8'
    assert model.base_url == 'https://inference.api.nscale.com/v1/'


@pytest.mark.anyio
async def test_nscale_glm_reasoning_content(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(
        completion_message(
            ChatCompletionMessage.model_construct(
                content='Here is the answer.',
                reasoning_content='I should answer directly.',
                role='assistant',
            )
        )
    )
    model = OpenAIChatModel(
        'glm-5.2-fp8',
        provider=NscaleProvider(openai_client=mock_client),
        profile=NscaleProvider.model_profile('glm/glm-5.2-fp8'),
    )

    response = await model.request(
        messages=[ModelRequest(parts=[UserPromptPart(content='Say hello')])],
        model_settings=ModelSettings(),
        model_request_parameters=ModelRequestParameters(),
    )

    assert response.parts == [
        ThinkingPart(content='I should answer directly.', id='reasoning_content', provider_name='nscale'),
        TextPart(content='Here is the answer.'),
    ]


@pytest.mark.anyio
async def test_nscale_glm_sends_required_tool_choice(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(
        completion_message(ChatCompletionMessage.model_construct(content='Done.', role='assistant'))
    )
    model = OpenAIChatModel(
        'glm-5.2-fp8',
        provider=NscaleProvider(openai_client=mock_client),
        profile=NscaleProvider.model_profile('glm/glm-5.2-fp8'),
    )

    await model.request(
        messages=[ModelRequest(parts=[UserPromptPart(content='Use the tool')])],
        model_settings=ModelSettings(tool_choice='required'),
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='lookup')],
            allow_text_output=True,
        ),
    )

    assert get_mock_chat_completion_kwargs(mock_client)[0]['tool_choice'] == 'required'


def test_nscale_glm_thinking_round_trip_mapping():
    model = OpenAIChatModel(
        'glm-5.2-fp8',
        provider=NscaleProvider(
            openai_client=MockOpenAI.create_mock(
                completion_message(ChatCompletionMessage.model_construct(content='', role='assistant'))
            )
        ),
        profile=NscaleProvider.model_profile('glm/glm-5.2-fp8'),
    )
    mapped = model._map_model_response(  # type: ignore[reportPrivateUsage]
        ModelResponse(
            parts=[
                ThinkingPart(content='Earlier reasoning.', id='reasoning_content', provider_name='nscale'),
                TextPart(content='Earlier answer.'),
            ]
        )
    )
    assert mapped == {
        'role': 'assistant',
        'reasoning_content': 'Earlier reasoning.',
        'content': 'Earlier answer.',
    }
