import shutil
from argparse import Namespace
import logging
import os
from inspect import isclass
from pathlib import Path

from PIL import (Image,
                 ImageFile,
                 ImageFilter,
                 )
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import models
from django.urls import reverse
from django.utils.timezone import now
from django.utils.translation import ugettext_lazy as _
from django.conf import settings

import cbir
import cbir.commands
from cbir.legacy_utils import find_image_files
from cbir.photo_storage_inverted_file import CBIR

logger = logging.getLogger('photologue.models')

# Default limit for gallery.latest
LATEST_LIMIT = 10

# Number of random images from the gallery to display.
SAMPLE_SIZE = 5

# max_length setting for the ImageModel ImageField
IMAGE_FIELD_MAX_LENGTH = 100

FILE_FIELD_MAX_LENGTH = 100

# Path to sample image
SAMPLE_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), 'res', 'sample.jpg')

# Modify image file buffer size.
ImageFile.MAXBLOCK = 256 * 2 ** 10

CONTENT_DIR_RELATIVE_TO_MEDIA_ROOT = 'content'  # relative to MEDIA_ROOT which is BASE_DIR + 'public/media/'
CONTENT_DIR = Path(settings.MEDIA_ROOT_RELATIVE_TO_BASE_DIR) / CONTENT_DIR_RELATIVE_TO_MEDIA_ROOT
CBIR_STATE_DIR = '.cbir'
DATABASE_ALL_PHOTOS = 'database_all'

####################################################################
# Support CACHEDIR.TAG spec for backups for ignoring cache dir.
# See http://www.brynosaurus.com/cachedir/spec.html
PHOTOLOGUE_CACHEDIRTAG = os.path.join(CONTENT_DIR_RELATIVE_TO_MEDIA_ROOT, "photos", "cache", "CACHEDIR.TAG")
if not default_storage.exists(PHOTOLOGUE_CACHEDIRTAG):
    default_storage.save(PHOTOLOGUE_CACHEDIRTAG, ContentFile(
        "Signature: 8a477f597d28d172789f06886806bc55"))

# Exif Orientation values
# Value 0thRow	0thColumn
#   1	top     left
#   2	top     right
#   3	bottom	right
#   4	bottom	left
#   5	left	top
#   6	right   top
#   7	right   bottom
#   8	left    bottom

# Image Orientations (according to EXIF informations) that needs to be
# transposed and appropriate action
IMAGE_EXIF_ORIENTATION_MAP = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    6: Image.ROTATE_270,
    8: Image.ROTATE_90,
}

# Quality options for JPEG images
JPEG_QUALITY_CHOICES = (
    (30, _('Very Low')),
    (40, _('Low')),
    (50, _('Medium-Low')),
    (60, _('Medium')),
    (70, _('Medium-High')),
    (80, _('High')),
    (90, _('Very High')),
)

# choices for new crop_anchor field in Photo
CROP_ANCHOR_CHOICES = (
    ('top', _('Top')),
    ('right', _('Right')),
    ('bottom', _('Bottom')),
    ('left', _('Left')),
    ('center', _('Center (Default)')),
)

IMAGE_TRANSPOSE_CHOICES = (
    ('FLIP_LEFT_RIGHT', _('Flip left to right')),
    ('FLIP_TOP_BOTTOM', _('Flip top to bottom')),
    ('ROTATE_90', _('Rotate 90 degrees counter-clockwise')),
    ('ROTATE_270', _('Rotate 90 degrees clockwise')),
    ('ROTATE_180', _('Rotate 180 degrees')),
)

WATERMARK_STYLE_CHOICES = (
    ('tile', _('Tile')),
    ('scale', _('Scale')),
)

# Prepare a list of image filters
filter_names = []
for n in dir(ImageFilter):
    klass = getattr(ImageFilter, n)
    if isclass(klass) and issubclass(klass, ImageFilter.BuiltinFilter) and \
            hasattr(klass, 'name'):
        filter_names.append(klass.__name__)
IMAGE_FILTERS_HELP_TEXT = _('Chain multiple filters using the following pattern "FILTER_ONE->FILTER_TWO->FILTER_THREE"'
                            '. Image filters will be applied in order. The following filters are available: %s.'
                            % (', '.join(filter_names)))

size_method_map = {}


def get_path_to_database(database, relative_to):
    """
    :param relative_to: ['base_dir', 'media_root']
    """
    options = ['base_dir', 'media_root']
    if relative_to not in options:
        raise ValueError(f'relative_to must be one of {options}. But {relative_to} given')

    prefix = CONTENT_DIR
    if relative_to == 'media_root':
        prefix = CONTENT_DIR_RELATIVE_TO_MEDIA_ROOT

    return str(Path(prefix) / database)


def get_storage_path_for_description_file_of_database(instance, filename):
    database = instance.slug
    return str(Path(get_path_to_database(database, relative_to='media_root')) / DATABASE_ALL_PHOTOS / 'database.txt')


def get_storage_path_for_description_file_of_event(instance, filename):
    database = instance.database.slug
    event = instance.slug
    return str(Path(get_path_to_database(database, relative_to='media_root')) / event / 'event.txt')


def get_storage_path_for_description_file_of_database_photo(instance, filename):
    database = instance.database.slug
    name, ext = filename.split()
    image_name, image_ext = instance.image.name.split()

    error_message_parts = []
    if ext != 'txt':
        error_message_parts += [f'Extension must be .txt not {ext}']
    if name != image_name:
        error_message_parts += [f'Description file name and image name must be equal: {name} != {image_name}']
    if error_message_parts:
        error_message = '-'.join(['Bad name for description'] + error_message_parts)
        logger.error(error_message)
        raise ValueError(error_message)

    fn = filename
    return str(Path(get_path_to_database(database, relative_to='media_root')) / DATABASE_ALL_PHOTOS / fn)


def get_storage_path_for_description_file_of_event_photo(instance, filename):
    database = instance.database.slug
    event = instance.slug
    name, ext = filename.split()
    image_name, image_ext = instance.image.name.split()

    error_message_parts = []
    if ext != 'txt':
        error_message_parts += [f'Extension must be .txt not {ext}']
    if name != image_name:
        error_message_parts += [f'Description file name and image name must be equal: {name} != {image_name}']
    if error_message_parts:
        error_message = '-'.join(['Bad name for description'] + error_message_parts)
        logger.error(error_message)
        raise ValueError(error_message)

    fn = filename
    return str(Path(get_path_to_database(database, relative_to='media_root')) / event / fn)


def get_storage_path_for_image(instance, filename):
    if isinstance(instance, DatabasePhoto):
        database = instance.database.slug
        folder = 'database_all'
    elif isinstance(instance, EventPhoto):
        database = instance.event.database.slug
        folder = instance.event.slug

    fn = filename

    return str(Path(get_path_to_database(database, relative_to='media_root')) / folder / fn)


####################################################################


class Database(models.Model):
    date_added = models.DateTimeField(_('date published'),
                                      default=now)
    title = models.CharField(_('title'),
                             max_length=250,
                             unique=True)
    slug = models.SlugField(_('slug'),
                            unique=True,
                            max_length=250,
                            help_text=_('A "slug" is a unique URL-friendly title for an object.'))
    description = models.TextField(_('description'),
                                   blank=True)
    description_file = models.FileField('description_file',
                                        max_length=FILE_FIELD_MAX_LENGTH,
                                        upload_to=get_storage_path_for_description_file_of_database,
                                        blank=True)
    cbir_index_default = models.OneToOneField(to='photologue.CbirIndex',

                                              # Note: not index related to database by default but
                                              # database has this index as default
                                              related_name='database_default',
                                              on_delete=models.SET_NULL,
                                              null=True,
                                              blank=True)

    class Meta:
        ordering = ['-date_added']
        get_latest_by = 'date_added'
        verbose_name = _('database')
        verbose_name_plural = _('databases')

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photologue:database_detail', args=[self.slug])

    def latest(self, limit=None):
        if not limit:
            limit = LATEST_LIMIT
        raise NotImplemented

    def get_events(self, limit=10):
        limit = limit or LATEST_LIMIT
        events = self.event_set.all()[:limit]
        logger.info(f'events: {events}')
        return events

    def save(self):
        if not self.description_file:
            logger.info('No description file')
            path = get_storage_path_for_description_file_of_database(self, filename=None)
            if not self.description:
                self.description = f'{self.title}\n{self.slug}'
            logger.info(f'Generate file: {path} with content: {self.description}')
            self.description_file.save(path, ContentFile(self.description))
        super().save()

    def get_name(self):
        return self.slug

    def get_path_to_all_photos(self):
        name = self.slug
        return str(Path(get_path_to_database(name, relative_to='base_dir')) / DATABASE_ALL_PHOTOS)

    def get_photo_by_name(self, name):
        database_photo = DatabasePhoto.objects.get(name=name, database=self)
        return database_photo


class CbirIndex(models.Model):
    DES_TYPE = 'l2net'
    L = 10
    K = 2
    MAX_KEYPOINTS = 200

    date_added = models.DateTimeField(_('date published'),
                                      default=now)
    title = models.CharField(_('title'),
                             max_length=250,
                             unique=True)
    slug = models.SlugField(_('slug'),
                            unique=True,
                            max_length=250,
                            help_text=_('A "slug" is a unique URL-friendly title for an object.'))
    name = models.CharField('name',
                            max_length=250,
                            help_text="Name. Directory with Index's data structures. "
                                      "must be stored at $CBIR_STATE_DIR/databases/$database/$name")
    description = models.TextField(_('description'),
                                   blank=True)
    database = models.ForeignKey(to='photologue.Database',
                                 on_delete=models.SET_NULL,
                                 null=True)
    count_photos_indexed = models.PositiveIntegerField('count_photos_indexed',
                                                       null=False,
                                                       blank=True,
                                                       default=0)
    built = models.BooleanField('built',
                                null=False,
                                blank=False,
                                default=False)

    class Meta:
        ordering = ['-date_added']
        get_latest_by = 'date_added'
        verbose_name = _('CBIR index')
        verbose_name_plural = _('CBIR indexes')

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photologue:database_index_detail', args=[self.slug])

    def build_if_needed(self):
        if self.building_needed:
            self.build()

    def building_needed(self):
        return not self.built and not self.being_built()

    def build(self):
        database_name = self.database.slug
        cbir_index_name = self.name

        path_to_database_photos = self.database.get_path_to_all_photos()
        list_paths_to_images_to_train_clusterer = find_image_files(path_to_database_photos, ['jpg'])
        list_paths_to_images_to_index = list_paths_to_images_to_train_clusterer

        CBIR.create_empty_if_needed(database_name, cbir_index_name,
                                    des_type=CbirIndex.DES_TYPE,
                                    max_keypoints=CbirIndex.MAX_KEYPOINTS,
                                    K=CbirIndex.K, L=CbirIndex.L)
        cbir_index = CBIR.get_instance(database_name, cbir_index_name)
        cbir_index.compute_descriptors(list(set(list_paths_to_images_to_index)
                                            | set(list_paths_to_images_to_train_clusterer)))
        cbir_index.train_clusterer(list_paths_to_images_to_train_clusterer)
        cbir_index.add_images_to_index(list_paths_to_images_to_index)

        self.built = True

    def being_built(self):
        return False


class Event(models.Model):
    date_added = models.DateTimeField(_('date published'),
                                      default=now)
    title = models.CharField(_('title'),
                             max_length=250,
                             unique=True)
    slug = models.SlugField(_('slug'),
                            unique=True,
                            max_length=250,
                            help_text=_('A "slug" is a unique URL-friendly title for an object.'))
    description = models.TextField(_('description'),
                                   blank=True)
    description_file = models.FileField('description_file',
                                        max_length=FILE_FIELD_MAX_LENGTH,
                                        upload_to=get_storage_path_for_description_file_of_event,
                                        blank=True)
    database = models.ForeignKey(to=Database, on_delete=models.CASCADE)

    cbir_index = models.ForeignKey(to=CbirIndex,
                                   on_delete=models.SET_NULL,
                                   null=True,
                                   blank=True)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photologue:event_detail', args=[self.slug])

    def save(self):
        if not self.description_file:
            path = get_storage_path_for_description_file_of_database(self, filename=None)
            if not self.description:
                self.description = f'{self.title}\n{self.slug}'
            self.description_file.save(path, ContentFile(self.description))
        super().save()

    def init_if_needed_and_get_result_photos(self):
        result_photos = self.get_result_photos()
        event_inited = len(result_photos) > 0
        if not event_inited:
            result_photos_names = self._do_search()
            self.set_result_photos_from_names(result_photos_names)
            result_photos = self.get_result_photos()
        return result_photos

    def set_result_photos_from_names(self, result_photos_names):
        database_photos_names = [Path(photo_name).name for photo_name in result_photos_names]
        for database_photo_name in database_photos_names:
            database_photo = self.database.get_photo_by_name(name=database_photo_name)

            event_photo = EventPhoto(slug='',
                                     event=self,
                                     is_query=False,
                                     database_photo=database_photo)
            event_photo.save()

    def get_result_photos(self):
        return EventPhoto.objects.filter(event=self).filter(is_query=False)

    def get_query_photos(self):
        return EventPhoto.objects.filter(event=self).filter(is_query=True)

    def _do_search(self):
        cbir_database_name = self.database.get_name()
        cbir_index_name = self.cbir_index.name

        query_photos = self.get_query_photos()
        if len(query_photos) == 0:
            logger.warning(f'Event {self} does not have query photos')
            return []

        query = str(Path(settings.MEDIA_ROOT_RELATIVE_TO_BASE_DIR) / query_photos[0].image.name)
        cbir_index = CBIR.get_instance(cbir_database_name, cbir_index_name)
        cbir_index.set_fd(cbir_index.load_fd())
        cbir_index.set_ca(cbir_index.load_ca())
        result_photos_names = cbir_index.search(query, qe_enable=False)
        result_photos_names = [v[1] for v in list(zip(*result_photos_names))[0]]
        cbir_index.unset_fd()
        cbir_index.unset_ca()

        return result_photos_names

    def has_cbir_index(self):
        logger.info(f'bool(self.cbir_index): {bool(self.cbir_index)}')
        return bool(self.cbir_index)

    def set_default_cbir_index_and_return_whether_success(self):
        default = self.database.cbir_index_default
        if default:
            self.cbir_index = default
            return True
        return False


class ImageModel(models.Model):
    image = models.ImageField('image',
                              max_length=IMAGE_FIELD_MAX_LENGTH,
                              upload_to=get_storage_path_for_image)


class DatabasePhoto(ImageModel):
    slug = models.SlugField('slug',
                            unique=True, )
    name = models.CharField('name',
                            max_length=250,
                            help_text='Name equal to corresponding filename',
                            unique=False,  # in different databases Photos can have equal names
                            null=False,
                            db_index=True)
    description = models.TextField('description',
                                   blank=True)

    # TODO: Find out whether description file is needed for database photo
    description_file = models.FileField('description_file',
                                        max_length=FILE_FIELD_MAX_LENGTH,
                                        upload_to=get_storage_path_for_description_file_of_database_photo,
                                        blank=True)

    database = models.ForeignKey(to=Database, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.slug} from {self.database}'

    def save(self):
        # TODO: Fix this strange saving.
        super().save()
        self.name = Path(self.image.name).name
        super().save()


class EventPhoto(ImageModel):
    slug = models.SlugField('slug',
                            unique=False, )
    description = models.TextField('description',
                                   blank=True)

    # TODO: Find out whether description file is needed for event photo
    description_file = models.FileField('description_file',
                                        max_length=FILE_FIELD_MAX_LENGTH,
                                        upload_to=get_storage_path_for_description_file_of_event_photo,
                                        blank=True)

    is_query = models.BooleanField('is_query',
                                   default=False, )
    event = models.ForeignKey(to=Event,
                              on_delete=models.CASCADE)
    database_photo = models.ForeignKey(to=DatabasePhoto,
                                       on_delete=models.SET_NULL,
                                       null=True)

    def __str__(self):
        return f'{self.database_photo.name} from {self.event}'

    def save(self):
        photo_path = get_storage_path_for_image(self,
                                                filename=Path(self.database_photo.image.name).name)
        self.image = photo_path
        path_to_original_file = os.path.join(settings.MEDIA_ROOT, self.database_photo.image.name)
        path_to_new_file = os.path.join(settings.MEDIA_ROOT, photo_path)
        shutil.copyfile(path_to_original_file, path_to_new_file)
        super().save()
