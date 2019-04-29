import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from django.core.management.base import BaseCommand, CommandError
from django.shortcuts import get_object_or_404
from django.conf import settings

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
                                   slug=database_name, )
        database.save()

        # Handling files
        logger.info('Handling files')
        count_files = 1
        list_paths_to_photos = find_image_files(directory, extensions=['jpg'], recursive=True)
        for path_to_photo in tqdm(list_paths_to_photos):
            while True:
                slug = f'{database.slug}-{count_files}'
                if models.DatabasePhoto.objects.filter(slug=slug).exists():
                    count_files += 1
                    continue
                break

            database_photo = models.DatabasePhoto(slug=slug,
                                                  database=database, )
            full_path_to_original_photo = os.path.join(settings.BASE_DIR, path_to_photo)
            database_photo.copy_photo_and_assign_image_field(full_path_to_original_photo)
            database_photo.name = Path(database_photo.image.name).name
            database_photo.save()
