import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.shortcuts import get_object_or_404
from photologue import models
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Search in database with index'

    def add_arguments(self, parser):
        parser.add_argument('database')
        parser.add_argument('index_name')
        parser.add_argument('query')
        parser.add_argument('--event_slug', required=False)
        parser.add_argument('--sv', action='store_true', default=False)
        parser.add_argument('--qe', action='store_true', default=False)


    def handle(self, *args, **options):
        database_name = options['database']
        cbir_index_name = options['index_name']
        query = options['query']
        event_slug = options['event_slug'] or 'event'
        if models.Event.objects.filter(slug=event_slug).exists():
            counter = 1
            while True:
                if models.Event.objects.filter(slug=event_slug + f'_{counter}').exists():
                    counter += 1
                else:
                    break
            event_slug = event_slug + f'_{counter}'

        database = get_object_or_404(models.Database, slug=database_name)
        cbir_index = get_object_or_404(models.CBIRIndex, database=database, name=cbir_index_name)

        event = models.Event(
            database=database,
            cbir_index=cbir_index,
            slug=event_slug,
            title=event_slug,
        )

        if event.cbir_index.database != database:
            # TODO: Fine that user's cbir_index is ignored silently? User is not told about it.
            logger.warning(f'User chose incorrect cbir_index connected to database {event.cbir_index.database}. '
                           f'But intends to create an event for {database} database')
            event.cbir_index = database.cbir_index_default
        if not event.cbir_index:
            event.cbir_index = database.cbir_index_default

        event.save()

        count_event_photo = 1
        while True:
            event_photo_slug = f'{event.slug}-{count_event_photo}'
            if models.EventPhoto.objects.filter(slug=event_photo_slug).exists():
                count_event_photo += 1
            else:
                break

        database_photo_slug_base = f'{database.slug}-{event_photo_slug}'
        count_database_photo = 1
        while True:
            database_photo_slug = f'{database_photo_slug_base}-{count_database_photo}'
            if models.DatabasePhoto.objects.filter(slug=database_photo_slug).exists():
                count_database_photo += 1
                continue
            break

        database_photo = models.DatabasePhoto(slug=database_photo_slug,
                                              database=database, )
        full_path_to_original_photo = os.path.join(settings.BASE_DIR, query)
        database_photo.copy_photo_and_assign_image_field(full_path_to_original_photo)
        database_photo.name = Path(database_photo.image.name).name

        database_photo.save()

        event_photo = models.EventPhoto(slug=event_photo_slug,
                                        event=event,
                                        is_query=True,
                                        database_photo=database_photo, )
        event_photo.save()


        sv = options['sv']
        qe = options['qe']
        n_candidates = 100
        topk = 5
        search_params = {
            'sv_enable': sv,
            'qe_enable': qe,
            'n_candidates': n_candidates,
            'topk': topk
        }
        result_photos = event.init_if_needed_and_get_result_photos(search_params=search_params)
