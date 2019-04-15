import os
from pathlib import Path


# Directory containing manage.py or main.py
ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERSISTENT_STATE = ROOT / '.cbir'
DATABASES = PERSISTENT_STATE / 'databases'
SESSIONS = PERSISTENT_STATE / 'sessions'  # TODO: Delete.
QUERIES = PERSISTENT_STATE / 'queries'  # TODO: Delete.

cuda_enabled = bool(os.environ.get('CUDA_HOME'))
CONFIG = {
    'cpu_required': not cuda_enabled,
    'use_cuda': cuda_enabled,
    'database': None,
}
