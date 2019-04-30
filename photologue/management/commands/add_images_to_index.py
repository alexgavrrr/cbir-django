import logging

from django.core.management.base import BaseCommand

from cbir.cbir_core import CBIRCore

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Add images to index: apply clusterer to every photo from database, build inverted index'

    def add_arguments(self, parser):
        parser.add_argument('database')
        parser.add_argument('index_name')

    def handle(self, *args, **options):
        database_name = options['database']
        cbir_index_name = options['index_name']

        cbir_core = CBIRCore.get_instance(database_name, cbir_index_name)
        # cbir_core.train_clusterer()
        cbir_core.add_images_to_index()
