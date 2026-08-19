import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import TestEnv, try_import
from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
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


@pytest.mark.parametrize(
    ('model_name', 'supports_thinking', 'thinking_always_enabled'),
    [
        ('moonshotai/Kimi-K2.5', True, False),
        ('openai/gpt-oss-120b', True, True),
        ('openai/gpt-oss-20b', True, True),
        ('Qwen/Qwen3-235B-A22B', True, True),
        ('Qwen/Qwen3-235B-A22B-Instruct-2507', False, False),
        ('Qwen/Qwen3-4B-Thinking-2507', False, False),
        ('Qwen/Qwen3-4B-Instruct-2507', False, False),
        ('Qwen/Qwen3-8B', True, True),
        ('Qwen/Qwen3-14B', True, True),
        ('Qwen/Qwen3-32B', True, True),
        ('meta-llama/Llama-4-Scout-17B-16E-Instruct', False, False),
        ('Qwen/Qwen2.5-Coder-3B-Instruct', False, False),
        ('Qwen/Qwen2.5-Coder-7B-Instruct', False, False),
        ('Qwen/Qwen2.5-Coder-32B-Instruct', False, False),
        ('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', True, False),
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', True, True),
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', True, True),
        ('mistralai/Devstral-Small-2505', False, False),
        ('mistralai/mixtral-8x22b-instruct-v0.1', False, False),
        ('meta-llama/Llama-3.1-8B-Instruct', False, False),
        ('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', True, True),
    ],
)
def test_nscale_external_chat_model_thinking_profiles(
    model_name: str, supports_thinking: bool, thinking_always_enabled: bool
):
    """External Nscale chat models should match the observed live `reasoning_effort` support matrix."""
    profile = NscaleProvider.model_profile(model_name)
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is supports_thinking
    assert profile.thinking_always_enabled is thinking_always_enabled


def test_nscale_gpt_oss_profile_supports_thinking():
    """Nscale-hosted gpt-oss models accept enabled reasoning and reject `reasoning_effort='none'`."""
    profile = NscaleProvider.model_profile('openai/gpt-oss-120b')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True


def test_nscale_nvidia_profile_supports_always_on_thinking():
    """Nscale-hosted NVIDIA Nemotron accepts enabled reasoning and rejects `reasoning_effort='none'`."""
    profile = NscaleProvider.model_profile('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True


def test_nscale_qwen_thinking_profile_does_not_support_native_thinking():
    """Nscale-hosted Qwen3 Thinking writes reasoning-like text in `content`, not native reasoning fields."""
    profile = NscaleProvider.model_profile('Qwen/Qwen3-4B-Thinking-2507')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is False
    assert profile.thinking_always_enabled is False


def test_nscale_qwen_profile_supports_always_on_thinking():
    """Base Qwen3 routes return native reasoning even with `reasoning_effort='none'`."""
    profile = NscaleProvider.model_profile('Qwen/Qwen3-8B')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True


def test_nscale_deepseek_llama_profile_supports_optional_thinking():
    """Nscale-hosted DeepSeek R1 Llama stops returning native reasoning when `reasoning_effort='none'`."""
    profile = NscaleProvider.model_profile('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is False


def test_nscale_qwen_instruct_profile_does_not_support_thinking():
    """Nscale-hosted Qwen3 Instruct routes should ignore the unified `thinking` setting."""
    profile = NscaleProvider.model_profile('Qwen/Qwen3-4B-Instruct-2507')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is False
    assert profile.thinking_always_enabled is False


def test_nscale_devstral_profile_does_not_support_thinking():
    """Nscale-hosted Devstral rejects enabled reasoning and should ignore the unified `thinking` setting."""
    profile = NscaleProvider.model_profile('mistralai/Devstral-Small-2505')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.supports_thinking is False
    assert profile.thinking_always_enabled is False


@pytest.mark.parametrize('model_name', ['glm-5.2-fp8', 'zai-org/GLM-5.2'])
def test_nscale_glm_model_profile(model_name: str):
    """Nscale-hosted GLM returns native reasoning, and `reasoning_effort='none'` disables it."""
    profile = NscaleProvider.model_profile(model_name)
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is False
    assert profile.openai_chat_thinking_field == 'reasoning_content'
    assert profile.openai_supports_tool_choice_required is False


def test_nscale_glm_model_inference(env: TestEnv):
    env.set('NSCALE_SERVICE_TOKEN', 'env-token')
    model = infer_model('nscale:glm-5.2-fp8')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'glm-5.2-fp8'
    assert model.base_url == 'https://inference.api.nscale.com/v1/'
    assert model.profile.supports_thinking is True


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
        profile=NscaleProvider.model_profile('glm-5.2-fp8'),
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
async def test_nscale_glm_rejects_required_tool_choice(allow_model_requests: None):
    """Nscale-hosted GLM rejects explicit forced tool choice."""
    mock_client = MockOpenAI.create_mock(
        completion_message(ChatCompletionMessage.model_construct(content='Done.', role='assistant'))
    )
    model = OpenAIChatModel(
        'glm-5.2-fp8',
        provider=NscaleProvider(openai_client=mock_client),
        profile=NscaleProvider.model_profile('glm-5.2-fp8'),
    )

    with pytest.raises(UserError, match="tool_choice='required' is not supported"):
        await model.request(
            messages=[ModelRequest(parts=[UserPromptPart(content='Use the tool')])],
            model_settings=ModelSettings(tool_choice='required'),
            model_request_parameters=ModelRequestParameters(
                function_tools=[ToolDefinition(name='lookup')],
                allow_text_output=True,
            ),
        )

    assert get_mock_chat_completion_kwargs(mock_client) == []


def test_nscale_glm_thinking_round_trip_mapping():
    model = OpenAIChatModel(
        'glm-5.2-fp8',
        provider=NscaleProvider(
            openai_client=MockOpenAI.create_mock(
                completion_message(ChatCompletionMessage.model_construct(content='', role='assistant'))
            )
        ),
        profile=NscaleProvider.model_profile('glm-5.2-fp8'),
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
