import logging
import logging.config
import os
from pathlib import Path

import cbir


def make_config(config_path, database, cbir_index_name, **kwargs):
    """
    :return config: {
        'database':,
        'cbir_index_name':,
    }
    :raises ValueError: "required parameter not specified"
    """
    raise NotImplemented


def configure_logging(
        log_level,
        prefix=None,
        **kwargs):

    prefix = prefix or 'logs/default/'
    root_profile = str(Path(prefix) / 'profile.txt')
    retrieving_candidates = str(Path(prefix) / 'retrieving_candidates.txt')
    preliminary_sorting = str(Path(prefix) / 'preliminary_sorting.txt')

    computing_descriptors = str(Path(prefix) / 'computing_descriptors.txt')
    training_clusterer = str(Path(prefix) / 'training_clusterer.txt')
    add_images_to_index = str(Path(prefix) / 'add_images_to_index.txt')


    if not os.path.exists(prefix):
        os.makedirs(prefix, exist_ok=True)

    log_level = log_level.upper()

    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                'format': '%(levelname)s %(name)s.%(funcName)s (%(lineno)d) %(message)s'
            }
        },
        'handlers': {
            'profile': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': root_profile
            },
            'profile.retrieving_candidates': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': retrieving_candidates
            },
            'profile.preliminary_sorting': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': preliminary_sorting
            },

            'profile.computing_descriptors': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': computing_descriptors
            },
            'profile.training_clusterer': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': training_clusterer
            },
            'profile.add_images_to_index': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': add_images_to_index
            },

            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'console'
            },
            'null': {
                'level': 'DEBUG',
                'class': 'logging.NullHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': log_level,
                'propagate': False,
            },
            'profile': {
                'handlers': ['profile'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'profile.retrieving_candidates': {
                'handlers': ['profile.retrieving_candidates'],
                'level': 'DEBUG',
                'propagate': True,
            },
            'profile.preliminary_sorting': {
                'handlers': ['profile.preliminary_sorting'],
                'level': 'DEBUG',
                'propagate': True,
            },

            'profile.computing_descriptors': {
                'handlers': ['profile.computing_descriptors'],
                'level': 'DEBUG',
                'propagate': True,
            },
            'profile.training_clusterer': {
                'handlers': ['profile.training_clusterer'],
                'level': 'DEBUG',
                'propagate': True,
            },
            'profile.add_images_to_index': {
                'handlers': ['profile.add_images_to_index'],
                'level': 'DEBUG',
                'propagate': True,
            },
        }
    }
    logging.config.dictConfig(log_config)
