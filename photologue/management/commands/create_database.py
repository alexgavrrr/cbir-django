import logging
import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from tqdm import tqdm

from cbir.legacy_utils import find_image_files
from photologue import models

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Create database'

    def add_arguments(self, parser):
        parser.add_argument('database')
        parser.add_argument('directory')

    def handle(self, *args, **options):
        database_name = options['database']
        directory = options['directory']

        database = models.Database(title=database_name,
                                   slug=database_name,
                                   count=0)
        database.save()

        # Handling files
        logger.info('Handling files')
        count_files = 0
        list_paths_to_photos = find_image_files(directory, extensions=['jpg'], recursive=True)
        print(f'list_paths_to_photos: {list_paths_to_photos}')
        for path_to_photo in tqdm(list_paths_to_photos):
            database_photo = models.DatabasePhoto(database=database, )
            full_path_to_original_photo = os.path.join(settings.BASE_DIR, path_to_photo)
            database_photo.copy_photo_and_assign_image_field(full_path_to_original_photo)
            database_photo.name = Path(database_photo.image.name).name
            database_photo.save()
            count_files += 1
        database.count = count_files
        database.save()
