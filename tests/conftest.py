"""
Pytest configuration and fixtures.

Mocks heavy dependencies that may not be available in test environment.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies before any test imports
MOCK_MODULES = [
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'bitsandbytes',
    'trl',
    'wandb',
    'vllm',
    'tinker',
    'tinker.types',
    'tinker.types.tensor_data',
    'tqdm',
    'chz',
    'tinker_cookbook',
    'tinker_cookbook.renderers',
    'tinker_cookbook.hyperparam_utils',
    'tinker_cookbook.utils',
    'tinker_cookbook.utils.ml_log',
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Setup mock renderer classes (need __init__ to accept tokenizer)
mock_renderers = sys.modules['tinker_cookbook.renderers']


class MockQwen3Renderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


mock_renderers.Qwen3Renderer = MockQwen3Renderer
mock_renderers.Renderer = type('Renderer', (), {})

# Setup mock hyperparam utils
mock_hparams = sys.modules['tinker_cookbook.hyperparam_utils']
mock_hparams.get_lr = lambda model_name, rank: 1e-4

# Setup mock ml_log
mock_ml_log = sys.modules['tinker_cookbook.utils.ml_log']
mock_ml_log.setup_logging = MagicMock(return_value=MagicMock())

# Setup mock TensorData
mock_tensor_data = sys.modules['tinker.types.tensor_data']
mock_tensor_data.TensorData = MagicMock()
