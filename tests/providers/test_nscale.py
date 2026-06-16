
import httpx
import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import OpenAIError

    from pydantic_ai.providers.nscale import NscaleProvider


def test_nscale_provider():
    provider = NscaleProvider(service_token='foobar')
    assert provider.name == 'nscale'
    assert provider.client.api_key == 'foobar'
