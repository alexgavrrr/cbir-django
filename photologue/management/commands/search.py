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

    def handle(self, *args, **options):
        database_name = options['database']
        cbir_index_name = options['index_name']
        query = options['query']

        raise NotImplementedError
