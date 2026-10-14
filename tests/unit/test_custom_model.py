"""Custom model metadata is fetched without downloading weights."""
import json
import pytest


def test_custom_model_uses_hub_parameter_count():
    from autotune.infrastructure.models.hub_metadata import resolve_model
    replies = iter([{'safetensors': {'total': 2000000000}, 'gated': False, 'cardData': {'license':'apache-2.0'}}, {'max_position_embeddings':8192}])
    model = resolve_model('owner/model', transport=lambda request, timeout: json.dumps(next(replies)).encode())
    assert model.parameter_count == 2000000000 and model.context_length == 8192


def test_custom_model_without_parameter_metadata_fails():
    from autotune.infrastructure.models.hub_metadata import resolve_model
    with pytest.raises(ValueError, match='parameter'):
        resolve_model('owner/looks-like-1B', transport=lambda *args: b'{}')


def test_custom_model_does_not_accept_urls():
    from autotune.infrastructure.models.hub_metadata import resolve_model
    with pytest.raises(ValueError, match='Hugging Face ID'):
        resolve_model('https://evil.test/model', transport=lambda *args: pytest.fail('no request'))


def test_custom_lookup_network_failure_is_actionable():
    from autotune.infrastructure.models.hub_metadata import resolve_model
    def offline(*args): raise OSError('offline')
    with pytest.raises(ValueError, match='Hub'):
        resolve_model('owner/model', transport=offline)
