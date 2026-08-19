from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the GitHub Models provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

class NscaleProvider(Provider[AsyncOpenAI]):
    """Provider for NScale Models API.

    NScale Models provides access to various AI models through an OpenAI-compatible API.
    See <https://docs.nscale.com/docs/ai-services/models> for more information.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...


    @property
    def name(self) -> str:
        return 'nscale'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'openai': openai_model_profile,
            'moonshotai': moonshotai_model_profile,
            'Qwen': qwen_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
            'nvidia': openai_model_profile,
            'zai-org': openai_model_profile,
        }
        profile: ModelProfile | None = None

        model_name = model_name.removeprefix('~')
        if '/' in model_name:
            provider, model_name = model_name.split('/', 1)
        elif model_name.startswith('glm-'):
            provider = 'zai-org'
        else:
            return None

        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)
        else:
            return
        profile = provider_to_profile[provider](model_name)
        is_gpt_oss = provider == 'openai' and model_name.startswith('gpt-oss')
        qwen3_always_reasoning_models = {
            'Qwen3-235B-A22B',
            'Qwen3-8B',
            'Qwen3-14B',
            'Qwen3-32B',
        }
        is_qwen3_always_reasoning = provider == 'Qwen' and model_name in qwen3_always_reasoning_models
        is_deepseek_r1 = provider == 'deepseek-ai' and model_name.startswith('DeepSeek-R1')
        deepseek_r1_always_reasoning_models = {
            'DeepSeek-R1-Distill-Qwen-7B',
            'DeepSeek-R1-Distill-Qwen-14B',
        }
        is_deepseek_r1_always_reasoning = provider == 'deepseek-ai' and model_name in deepseek_r1_always_reasoning_models
        is_kimi_reasoning = provider == 'moonshotai' and model_name == 'Kimi-K2.5'
        is_nvidia_reasoning = provider == 'nvidia'
        supports_thinking = (
            is_gpt_oss
            or is_qwen3_always_reasoning
            or is_deepseek_r1
            or is_kimi_reasoning
            or is_nvidia_reasoning
        )
        thinking_always_enabled = (
            is_gpt_oss
            or is_qwen3_always_reasoning
            or is_deepseek_r1_always_reasoning
            or is_nvidia_reasoning
        )

        for prefix, profile_func in provider_to_profile.items():
            if model_name.startswith(prefix):
                profile = profile_func(model_name)
                break

        if provider == 'zai-org':
            nscale_profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_tool_choice_required=False,
                supports_thinking=True,
                thinking_always_enabled=False,
                openai_chat_thinking_field='reasoning_content',
                openai_chat_send_back_thinking_parts='auto',
            )
        elif provider in ('deepseek-ai', 'mistralai', 'Qwen'):
            nscale_profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_tool_choice_required=False,
                supports_tools=False,
                default_structured_output_mode='prompted',
                supports_thinking=supports_thinking,
                thinking_always_enabled=thinking_always_enabled,
            )
        else:
            nscale_profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_tool_choice_required=False,
                supports_thinking=supports_thinking,
                thinking_always_enabled=thinking_always_enabled,
            )                    

        return nscale_profile.update(profile)
    

    def __init__(
        self,
        base_url: str | None =None,
        service_token: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Nscale provider.
            Args:

        Args:
            service_token: Scale service Token. If not provided, reads from NSCALE_SERVICE_TOKEN env var.
            base_url: Custom API base URL. Defaults to https://inference.api.nscale.com/v1
            openai_client: Optional pre-configured OpenAI client
            http_client: Optional custom httpx.AsyncClient for making HTTP requests

        Raises:
            UserError: If API key is not provided and NSCALE_SERVICE_TOKEN env var is not set
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            # Get API key from parameter or environment
            service_token = service_token or os.getenv('NSCALE_SERVICE_TOKEN')
            if not service_token:
                raise UserError(
                    'Set the `NSCALE_SERVICE_TOKEN` environment variable or pass it via '
                    '`NScaleProvider(service_token=...)` to use the NScale provider.'
                )

            # Set base URL (default to NScale API endpoint)
            self._base_url = base_url or os.getenv('NSCALE_BASE_URL', 'https://inference.api.nscale.com/v1')

            if http_client is None:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=service_token, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
