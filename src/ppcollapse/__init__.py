__version__ = "0.1.0"

import logging

# Import modules
from ppcollapse import (
    utils,  # noqa: F401
)

# Import specific functions
from ppcollapse.utils.logger import (  # noqa: F401
    get_logger,
    set_log_level,
    setup_logger,
)

logger = setup_logger(logging.INFO, name="ppcx", force=True)
