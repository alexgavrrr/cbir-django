import logging
import os
import random
import unicodedata
from datetime import datetime
from inspect import isclass
from io import BytesIO

import exifread
from PIL import (Image,
                 ImageFile,
                 ImageFilter,
                 )
from django.conf import settings
from django.contrib.sites.models import Site
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import models
from django.template.defaultfilters import slugify
from django.urls import reverse
from django.utils.encoding import force_text, smart_str, filepath_to_uri
from django.utils.functional import curry
from django.utils.safestring import mark_safe
from django.utils.timezone import now
from django.utils.translation import ugettext_lazy as _
from sortedm2m.fields import SortedManyToManyField


logger = logging.getLogger('image_storage.models')

# Default limit for gallery.latest
LATEST_LIMIT = None

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

CONTENT_DIR = 'content'

####################################################################
# Support CACHEDIR.TAG spec for backups for ignoring cache dir.
# See http://www.brynosaurus.com/cachedir/spec.html
PHOTOLOGUE_CACHEDIRTAG = os.path.join(CONTENT_DIR, "photos", "cache", "CACHEDIR.TAG")
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


# def get_storage_path(instance, filename):
#     fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
#     database = 'photos'
#     return os.path.join(CONTENT_DIR, database, fn)

def get_storage_path_for_description_file_of_database(instance, filename):
    fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
    database = instance.slug
    return os.path.join(CONTENT_DIR, database, fn)

def get_storage_path_for_description_file_of_event(instance, filename):
    fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
    database = instance.database.slug
    event = instance.slug
    return os.path.join(CONTENT_DIR, database, event, fn)

def get_storage_path_for_description_file_of_database_photo(instance, filename):
    fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
    database = instance.database.slug
    folder = f'{database}_all'
    return os.path.join(CONTENT_DIR, database, folder, fn)

def get_storage_path_for_description_file_of_event_photo(instance, filename):
    fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
    database = instance.database.slug
    event = instance.slug
    return os.path.join(CONTENT_DIR, database, event, fn)

def get_storage_path_for_image(instance, filename):
    # fn = unicodedata.normalize('NFKD', force_text(filename)).encode('ascii', 'ignore').decode('ascii')
    database = instance.database.slug
    folder = f'{database}_all'
    return os.path.join(CONTENT_DIR, database, folder, filename)


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

    # events


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

        # TODO: return self.events
        return []


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

    # query = models.ImageField('query',
    #                           max_length=IMAGE_FIELD_MAX_LENGTH,
    #                           upload_to=get_storage_path_for_query_of_event,
    #                           blank=True)

    def __str__(self):
        return self.title


class ImageModel(models.Model):
    image = models.ImageField('image',
                              max_length=IMAGE_FIELD_MAX_LENGTH,
                              upload_to=get_storage_path_for_image)


class Photo(ImageModel):
    title = models.CharField(_('title'),
                             max_length=250,
                             unique=True)
    slug = models.SlugField(_('slug'),
                            unique=True,
                            max_length=250,
                            help_text=_('A "slug" is a unique URL-friendly title for an object.'))
    caption = models.TextField(_('caption'),
                               blank=True)
    date_added = models.DateTimeField(_('date added'),
                                      default=now)




    class Meta:
        ordering = ['-date_added']
        get_latest_by = 'date_added'
        verbose_name = _("photo")
        verbose_name_plural = _("photos")

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # TODO
        if self.slug is None:
            self.slug = slugify(self.title)
        super(Photo, self).save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('photologue:pl-photo', args=[self.slug])


class DatabasePhoto(ImageModel):
    # title is not needed
    slug = models.SlugField('slug',
                            unique=True,)
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


class EventPhoto(ImageModel):
    # title is not needed
    slug = models.SlugField('slug',
                            unique=False,)
    description = models.TextField('description',
                                   blank=True)

    # TODO: Find out whether description file is needed for database photo
    description_file = models.FileField('description_file',
                                        max_length=FILE_FIELD_MAX_LENGTH,
                                        upload_to=get_storage_path_for_description_file_of_event_photo,
                                        blank=True)

    is_query = models.BooleanField('is_query',
                                   default=False,)
    event = models.ForeignKey(to=Event,
                              on_delete=models.CASCADE)
    database_photo = models.ForeignKey(to=DatabasePhoto,
                                       on_delete=models.SET_NULL,
                                       null=True)

    def __str__(self):
        return f'{self.slug} from {self.event}'
