"""Shared pytest fixtures.

Guarantees that a test which initializes the torch distributed process group
(e.g. the in-process training-loop tests) does not leave it alive for the rest
of the session: a lingering group makes torchmetrics try to all-reduce and
fails unrelated tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _clean_distributed_state():
    yield
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:  # pragma: no cover - best-effort hygiene
        pass
