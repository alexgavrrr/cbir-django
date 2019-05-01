from django.core.management.base import BaseCommand, CommandError
from django.shortcuts import get_object_or_404
from photologue import models
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Create index for database using another dataset for training'

    def add_arguments(self, parser):
        parser.add_argument('database')
        parser.add_argument('index_name')
        parser.add_argument('dataset_directory')
        parser.add_argument('--use_database_photos_for_training', action='store_true')

    def handle(self, *args, **options):
        database_name = options['database']
        cbir_index_name = options['index_name']
        dataset_directory = options['dataset_directory']
        use_database_photos_for_training = options['use_database_photos_for_training']

        database = get_object_or_404(models.Database, slug=database_name)

        # cbir_index = models.CBIRIndex(
        #     name=cbir_index_name,
        #     database=database,
        #     slug=cbir_index_name,
        #     title=cbir_index_name)

        cbir_index, created = models.CBIRIndex.objects.get_or_create(database=database, name=cbir_index_name)

        logger.info('Saved not yet built index')
        cbir_index.build_using_dataset_for_training(dataset_directory=dataset_directory,
                                                    use_database_photos_for_training=use_database_photos_for_training)
        logger.info('Built index')
