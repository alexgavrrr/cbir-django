import os
from pathlib import Path


# Directory containing manage.py or main.py
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERSISTENT_STATE = BASE_DIR / '.cbir'
DATABASES = PERSISTENT_STATE / 'databases'
DATABASES_RELATIVE_TO_BASE_DIR = str(Path(DATABASES).relative_to(BASE_DIR))

cuda_enabled = bool(os.environ.get('CUDA_HOME'))
CONFIG = {
    'cpu_required': not cuda_enabled,
    'use_cuda': cuda_enabled,
}
