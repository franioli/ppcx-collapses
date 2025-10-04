import numpy as np
from sqlalchemy import create_engine

from ppcollapse.utils.config import ConfigManager

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Load configuration
config = ConfigManager(config_path="config.yaml")
db_engine = create_engine(
    "postgresql://postgres:postgresppcx@150.145.51.193:5434/sandbox"
)

print(config)
