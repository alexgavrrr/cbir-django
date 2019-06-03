import logging

from django.core.management.base import BaseCommand
from django.shortcuts import get_object_or_404
from django.core.exceptions import ObjectDoesNotExist

from photologue import models

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Create index for database using another dataset for training'

    def add_arguments(self, parser):
        parser.add_argument('database')
        parser.add_argument('index_name')
        parser.add_argument('dataset_directory')
        parser.add_argument('--use_database_photos_for_training', action='store_true')
        parser.add_argument('--K', type=int, required=True)
        parser.add_argument('--L', type=int, required=True)
        parser.add_argument('--des_type', required=True)



    def handle(self, *args, **options):
        database_name = options['database']
        cbir_index_name = options['index_name']
        dataset_directory = options['dataset_directory']
        K = options['K']
        L = options['L']
        des_type = options['des_type']

        use_database_photos_for_training = options['use_database_photos_for_training']

        database = get_object_or_404(models.Database, slug=database_name)

        cbir_index = None
        try:
            cbir_index = models.CBIRIndex.objects.get(database=database, name=cbir_index_name)
            logger.info('Already exists')
            # print('Exiting')
            # # TODO maybe enforce building
            # return
        except ObjectDoesNotExist:
            cbir_index = models.CBIRIndex(database=database,
                                         name=cbir_index_name,
                                         slug=cbir_index_name,
                                         title=cbir_index_name,
                                         count_photos_indexed=0,
                                         count_photos_for_training_from_database=0,
                                         count_photos_for_training_external=0,
                                         )
        build_params = {
            'K': K,
            'L': L,
            'des_type': des_type,
        }

        logger.info('Saved not yet built index')
        cbir_index.build_using_dataset_for_training(dataset_directory=dataset_directory,
                                                    use_database_photos_for_training=use_database_photos_for_training,
                                                    build_params=build_params)
        logger.info('Finished building index')
