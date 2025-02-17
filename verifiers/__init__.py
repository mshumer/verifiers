from .envs.base import BaseEnv
from .envs.code_env import CodeEnv
from .envs.doublecheck_env import DoubleCheckEnv
from .envs.math_env import MathEnv
from .envs.simple_env import SimpleEnv
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, preprocess_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import get_default_grpo_config
from .utils.logging_utils import setup_logging
from .judges import llm_judge_reward_func

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "BaseEnv",
    "CodeEnv",
    "DoubleCheckEnv",
    "MathEnv",
    "SimpleEnv",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "llm_judge_reward_func",
]