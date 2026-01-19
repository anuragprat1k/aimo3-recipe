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
    'chz',
    'tinker_cookbook',
    'tinker_cookbook.abstractions',
    'tinker_cookbook.abstractions.renderers',
    'tinker_cookbook.abstractions.hparams',
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Setup mock renderer classes to return proper mock instances
mock_renderers = sys.modules['tinker_cookbook.abstractions.renderers']
mock_renderers.Llama3ChatRenderer = type('Llama3ChatRenderer', (), {})
mock_renderers.Qwen2ChatRenderer = type('Qwen2ChatRenderer', (), {})

# Setup mock hparams
mock_hparams = sys.modules['tinker_cookbook.abstractions.hparams']
mock_hparams.compute_lora_learning_rate = lambda rank: 1e-4
