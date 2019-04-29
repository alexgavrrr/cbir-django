import shutil
from argparse import Namespace
import logging
import os
from datetime import datetime
from inspect import isclass
from io import BytesIO
from pathlib import Path

import exifread
from PIL import (Image,
                 ImageFile,
                 ImageFilter,
                 ImageEnhance,)
from django.core.exceptions import ValidationError

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.validators import RegexValidator
from django.db import models
from django.urls import reverse
from django.utils.encoding import force_text, filepath_to_uri, smart_str
from django.utils.functional import curry
from django.utils.safestring import mark_safe
from django.utils.timezone import now
from django.utils.translation import ugettext_lazy as _
from django.conf import settings

from .utils.reflection import add_reflection
from .utils.watermark import apply_watermark

import cbir
import cbir.commands
from cbir.legacy_utils import find_image_files
from cbir.cbir_core import CBIRCore

logger = logging.getLogger('photologue.models')

# Default limit for gallery.latest
LIMIT_PHOTOS = 10

# Number of random images from the gallery to display.
SAMPLE_SIZE = LIMIT_PHOTOS

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

PHOTOLOGUE_DIR = CONTENT_DIR_RELATIVE_TO_MEDIA_ROOT

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
    cbir_index_default = models.OneToOneField(to='photologue.CBIRIndex',

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

    def sample_photos(self, limit=None):
        if not limit:
            limit = LIMIT_PHOTOS
        return self.databasephoto_set.all()[:limit]

    def latest(self, limit=None):
        if not limit:
            limit = LIMIT_PHOTOS
        raise NotImplemented

    def get_events(self, limit=10):
        limit = limit or LIMIT_PHOTOS
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


class CBIRIndex(models.Model):
    # DES_TYPE = 'l2net'
    DES_TYPE = 'HardNetHPatches'
    L = 5
    K = 10
    MAX_KEYPOINTS = 2000

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
        verbose_name = _('CBIRCore index')
        verbose_name_plural = _('CBIRCore indexes')

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photologue:database_index_detail', args=[self.slug])

    def build_if_needed(self):
        if self.building_needed():
            self.build()

    def building_needed(self):
        return not self.built and not self.being_built()

    def build(self):
        database_name = self.database.slug
        cbir_index_name = self.name

        path_to_database_photos = self.database.get_path_to_all_photos()
        list_paths_to_images_to_train_clusterer = find_image_files(path_to_database_photos, ['jpg'], recursive=False)
        list_paths_to_images_to_index = list_paths_to_images_to_train_clusterer

        CBIRCore.create_empty_if_needed(database_name, cbir_index_name,
                                        des_type=CBIRIndex.DES_TYPE,
                                        max_keypoints=CBIRIndex.MAX_KEYPOINTS,
                                        K=CBIRIndex.K, L=CBIRIndex.L)
        cbir_core_index = CBIRCore.get_instance(database_name, cbir_index_name)
        cbir_core_index.compute_descriptors(list(set(list_paths_to_images_to_index)
                                                 | set(list_paths_to_images_to_train_clusterer)),
                                            to_index=True,
                                            for_training_clusterer=True)
        cbir_core_index.train_clusterer()
        cbir_core_index.add_images_to_index()
        self.built = True

    def being_built(self):
        # TODO: Delete it or begin using it, for example, for async call.
        return False

    def add_not_yet_indexed_photos(self):
        """
        Calls cbir_index_core's function to compute descriptors for photos from a database for which descriptors
        have not been computed yet. After calls cbir_index_core's function to add_photos_to_index.
        Before calling cbir_index_core's functions cbir_index(=self) should find out which photos it has not indexed yet.
        """
        # Define which photos have not been indexed yet by this index.
        # TODO: How? CBIRIndex should store information about which photos from database it has indexed.
        # new ones = Database photos - indexed
        # ManyToManyRelation?

        # cbir_instance.compute_descriptors(list_paths, to_index=True, for_training_clusterer=False)
        # cbir_instance.add_photos_to_index()

        raise NotImplementedError

    def build_index_based_on_other(self, cbir_index_to_copy):
        """

        :param database:
        :param cbir_index_to_copy:
        :return:
        """
        CBIRCore.create_empty_if_needed(self.database, self.name)
        cbir_core_index = CBIRCore.get_instance(self.database, self.name)

        for_training = True
        to_index = cbir_index_to_copy.database == self.database
        cbir_core_index.copy_descriptors_from_to(
            from_database=cbir_index_to_copy.database,
            from_name=cbir_index_to_copy.name,
            to_database=self.database,
            to_name=self.name,
            to_index=to_index,
            for_training=for_training)

        list_paths_to_photos_from_database_whose_descriptors_not_computed_yet = None  # TODO
        cbir_core_index.compute_descriptors(list_paths_to_photos_from_database_whose_descriptors_not_computed_yet,
                                            to_index=True,
                                            for_training_clusterer=True)
        cbir_core_index.train_clusterer()
        cbir_core_index.add_images_to_index()
        self.built = True

        raise NotImplementedError('list_paths_to_photos_from_database_whose_descriptors_not_computed_yet must find')


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

    cbir_index = models.ForeignKey(to=CBIRIndex,
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
            result_photos_names, result_photos_similarities = self._do_search()
            self.set_result_photos_from_names(result_photos_names, result_photos_similarities)
            result_photos = self.get_result_photos()
        return result_photos

    def set_result_photos_from_names(self, result_photos_names, result_photos_similarities):
        database_photos_names = [Path(photo_name).name for photo_name in result_photos_names]
        for database_photo_name, result_photo_similarity in zip(database_photos_names, result_photos_similarities):
            print(f'database_photo_name: {database_photo_name}')
            database_photo = self.database.get_photo_by_name(name=database_photo_name)
            event_photo = EventPhoto(slug='',
                                     event=self,
                                     is_query=False,
                                     database_photo=database_photo,
                                     similarity=result_photo_similarity)
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
        cbir_index = CBIRCore.get_instance(cbir_database_name, cbir_index_name)
        result_photos_raw = cbir_index.search(query, qe_enable=True)
        print(f'result_photos: {result_photos_raw}')
        print(f'result_photos[0]: {result_photos_raw[0]}')
        result_photos_names = [t[0][1] for t in result_photos_raw]
        result_photos_similarities = [t[1] for t in result_photos_raw]
        return result_photos_names, result_photos_similarities

    def has_cbir_index(self):
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

    date_taken = models.DateTimeField(_('date taken'),
                                      null=True,
                                      blank=True,
                                      help_text=_('Date image was taken; is obtained from the image EXIF data.'))
    view_count = models.PositiveIntegerField(_('view count'),
                                             default=0,
                                             editable=False)
    crop_from = models.CharField(_('crop from'),
                                 blank=True,
                                 max_length=10,
                                 default='center',
                                 choices=CROP_ANCHOR_CHOICES)
    effect = models.ForeignKey('photologue.PhotoEffect',
                               null=True,
                               blank=True,
                               related_name="%(class)s_related",
                               verbose_name=_('effect'),
                               on_delete=models.CASCADE)

    class Meta:
        abstract = True

    def copy_photo_and_assign_image_field(self, full_path_to_original_photo):
        path_to_new_photo = get_storage_path_for_image(
            self,
            filename=Path(full_path_to_original_photo).name)

        full_path_to_new_photo = os.path.join(settings.MEDIA_ROOT, path_to_new_photo)
        shutil.copyfile(full_path_to_original_photo, full_path_to_new_photo)
        self.image = path_to_new_photo

    def EXIF(self, file=None):
        try:
            if file:
                tags = exifread.process_file(file)
            else:
                with self.image.storage.open(self.image.name, 'rb') as file:
                    tags = exifread.process_file(file, details=False)
            return tags
        except:
            return {}

    def admin_thumbnail(self):
        func = getattr(self, 'get_admin_thumbnail_url', None)
        if func is None:
            return _('An "admin_thumbnail" photo size has not been defined.')
        else:
            if hasattr(self, 'get_absolute_url'):
                return mark_safe(u'<a href="{}"><img src="{}"></a>'.format(self.get_absolute_url(), func()))
            else:
                return mark_safe(u'<a href="{}"><img src="{}"></a>'.format(self.image.url, func()))
    admin_thumbnail.short_description = _('Thumbnail')
    admin_thumbnail.allow_tags = True

    def cache_path(self):
        # TODOATTEN
        return os.path.join(os.path.dirname(self.image.name), "cache")

    def cache_url(self):
        # TODOATTEN
        return '/'.join([os.path.dirname(self.image.url), "cache"])

    def image_filename(self):
        # TODOATTEN
        return os.path.basename(force_text(self.image.name))

    def _get_filename_for_size(self, size):
        size = getattr(size, 'name', size)
        base, ext = os.path.splitext(self.image_filename())
        return ''.join([base, '_', size, ext])

    def _get_SIZE_photosize(self, size):
        return PhotoSizeCache().sizes.get(size)

    def _get_SIZE_size(self, size):
        photosize = PhotoSizeCache().sizes.get(size)
        if not self.size_exists(photosize):
            self.create_size(photosize)
        try:
            return Image.open(self.image.storage.open(
            self._get_SIZE_filename(size))).size
        except:
            return None

    def _get_SIZE_url(self, size):
        photosize = PhotoSizeCache().sizes.get(size)
        if not self.size_exists(photosize):
            self.create_size(photosize)
        if photosize.increment_count:
            self.increment_count()
        return '/'.join([
            self.cache_url(),
            filepath_to_uri(self._get_filename_for_size(photosize.name))])

    def _get_SIZE_filename(self, size):
        photosize = PhotoSizeCache().sizes.get(size)
        return smart_str(os.path.join(self.cache_path(),
                                      self._get_filename_for_size(photosize.name)))

    def increment_count(self):
        self.view_count += 1
        models.Model.save(self)

    def __getattr__(self, name):
        global size_method_map
        if not size_method_map:
            init_size_method_map()
        di = size_method_map.get(name, None)
        if di is not None:
            result = curry(getattr(self, di['base_name']), di['size'])
            setattr(self, name, result)
            return result
        else:
            raise AttributeError

    def size_exists(self, photosize):
        func = getattr(self, "get_%s_filename" % photosize.name, None)
        if func is not None:
            if self.image.storage.exists(func()):
                return True
        return False

    def resize_image(self, im, photosize):
        cur_width, cur_height = im.size
        new_width, new_height = photosize.size
        if photosize.crop:
            ratio = max(float(new_width) / cur_width, float(new_height) / cur_height)
            x = (cur_width * ratio)
            y = (cur_height * ratio)
            xd = abs(new_width - x)
            yd = abs(new_height - y)
            x_diff = int(xd / 2)
            y_diff = int(yd / 2)
            if self.crop_from == 'top':
                box = (int(x_diff), 0, int(x_diff + new_width), new_height)
            elif self.crop_from == 'left':
                box = (0, int(y_diff), new_width, int(y_diff + new_height))
            elif self.crop_from == 'bottom':
                # y - yd = new_height
                box = (int(x_diff), int(yd), int(x_diff + new_width), int(y))
            elif self.crop_from == 'right':
                # x - xd = new_width
                box = (int(xd), int(y_diff), int(x), int(y_diff + new_height))
            else:
                box = (int(x_diff), int(y_diff), int(x_diff + new_width), int(y_diff + new_height))
            im = im.resize((int(x), int(y)), Image.ANTIALIAS).crop(box)
        else:
            if not new_width == 0 and not new_height == 0:
                ratio = min(float(new_width) / cur_width,
                            float(new_height) / cur_height)
            else:
                if new_width == 0:
                    ratio = float(new_height) / cur_height
                else:
                    ratio = float(new_width) / cur_width
            new_dimensions = (int(round(cur_width * ratio)),
                              int(round(cur_height * ratio)))
            if new_dimensions[0] > cur_width or \
               new_dimensions[1] > cur_height:
                if not photosize.upscale:
                    return im
            im = im.resize(new_dimensions, Image.ANTIALIAS)
        return im

    def create_size(self, photosize):
        if self.size_exists(photosize):
            return
        try:
            im = Image.open(self.image.storage.open(self.image.name))
        except IOError:
            return
        # Save the original format
        im_format = im.format
        # Apply effect if found
        if self.effect is not None:
            im = self.effect.pre_process(im)
        elif photosize.effect is not None:
            im = photosize.effect.pre_process(im)
        # Rotate if found & necessary
        if 'Image Orientation' in self.EXIF() and \
                self.EXIF().get('Image Orientation').values[0] in IMAGE_EXIF_ORIENTATION_MAP:
            im = im.transpose(
                IMAGE_EXIF_ORIENTATION_MAP[self.EXIF().get('Image Orientation').values[0]])
        # Resize/crop image
        if im.size != photosize.size and photosize.size != (0, 0):
            im = self.resize_image(im, photosize)
        # Apply watermark if found
        if photosize.watermark is not None:
            im = photosize.watermark.post_process(im)
        # Apply effect if found
        if self.effect is not None:
            im = self.effect.post_process(im)
        elif photosize.effect is not None:
            im = photosize.effect.post_process(im)
        # Save file
        im_filename = getattr(self, "get_%s_filename" % photosize.name)()
        try:
            buffer = BytesIO()
            # Issue #182 - test fix from https://github.com/bashu/django-watermark/issues/31
            if im.mode.endswith('A'):
                im = im.convert(im.mode[:-1])
            if im_format != 'JPEG':
                im.save(buffer, im_format)
            else:
                im.save(buffer, 'JPEG', quality=int(photosize.quality),
                        optimize=True)
            buffer_contents = ContentFile(buffer.getvalue())
            self.image.storage.save(im_filename, buffer_contents)
        except IOError as e:
            if self.image.storage.exists(im_filename):
                self.image.storage.delete(im_filename)
            raise e

    def remove_size(self, photosize, remove_dirs=True):
        if not self.size_exists(photosize):
            return
        filename = getattr(self, "get_%s_filename" % photosize.name)()
        if self.image.storage.exists(filename):
            self.image.storage.delete(filename)

    def clear_cache(self):
        cache = PhotoSizeCache()
        for photosize in cache.sizes.values():
            self.remove_size(photosize, False)

    def pre_cache(self):
        cache = PhotoSizeCache()
        for photosize in cache.sizes.values():
            if photosize.pre_cache:
                self.create_size(photosize)

    def __init__(self, *args, **kwargs):
        super(ImageModel, self).__init__(*args, **kwargs)
        self._old_image = self.image

    def save(self, *args, **kwargs):
        image_has_changed = False
        if self._get_pk_val() and (self._old_image != self.image):
            image_has_changed = True
            # If we have changed the image, we need to clear from the cache all instances of the old
            # image; clear_cache() works on the current (new) image, and in turn calls several other methods.
            # Changing them all to act on the old image was a lot of changes, so instead we temporarily swap old
            # and new images.
            new_image = self.image
            self.image = self._old_image
            self.clear_cache()
            self.image = new_image  # Back to the new image.
            self._old_image.storage.delete(self._old_image.name)  # Delete (old) base image.
        if self.date_taken is None or image_has_changed:
            # Attempt to get the date the photo was taken from the EXIF data.
            try:
                exif_date = self.EXIF(self.image.file).get('EXIF DateTimeOriginal', None)
                if exif_date is not None:
                    d, t = exif_date.values.split()
                    year, month, day = d.split(':')
                    hour, minute, second = t.split(':')
                    self.date_taken = datetime(int(year), int(month), int(day),
                                               int(hour), int(minute), int(second))
            except:
                logger.error('Failed to read EXIF DateTimeOriginal', exc_info=True)
        super(ImageModel, self).save(*args, **kwargs)
        self.pre_cache()

    def delete(self):
        assert self._get_pk_val() is not None, \
            "%s object can't be deleted because its %s attribute is set to None." % \
            (self._meta.object_name, self._meta.pk.attname)
        self.clear_cache()
        # Files associated to a FileField have to be manually deleted:
        # https://docs.djangoproject.com/en/dev/releases/1.3/#deleting-a-model-doesn-t-delete-associated-files
        # http://haineault.com/blog/147/
        # The data loss scenarios mentioned in the docs hopefully do not apply
        # to Photologue!
        super(ImageModel, self).delete()
        self.image.storage.delete(self.image.name)


class BaseEffect(models.Model):
    name = models.CharField(_('name'),
                            max_length=30,
                            unique=True)
    description = models.TextField(_('description'),
                                   blank=True)

    class Meta:
        abstract = True

    def sample_dir(self):
        # TODOATTEN: change `samples`?
        return os.path.join(PHOTOLOGUE_DIR, 'samples')

    def sample_url(self):
        return settings.MEDIA_URL + '/'.join([PHOTOLOGUE_DIR, 'samples', '%s %s.jpg' % (self.name.lower(), 'sample')])

    def sample_filename(self):
        return os.path.join(self.sample_dir(), '%s %s.jpg' % (self.name.lower(), 'sample'))

    def create_sample(self):
        try:
            im = Image.open(SAMPLE_IMAGE_PATH)
        except IOError:
            raise IOError(
                'Photologue was unable to open the sample image: %s.' % SAMPLE_IMAGE_PATH)
        im = self.process(im)
        buffer = BytesIO()
        # Issue #182 - test fix from https://github.com/bashu/django-watermark/issues/31
        if im.mode.endswith('A'):
            im = im.convert(im.mode[:-1])
        im.save(buffer, 'JPEG', quality=90, optimize=True)
        buffer_contents = ContentFile(buffer.getvalue())
        default_storage.save(self.sample_filename(), buffer_contents)

    def admin_sample(self):
        return u'<img src="%s">' % self.sample_url()
    admin_sample.short_description = 'Sample'
    admin_sample.allow_tags = True

    def pre_process(self, im):
        return im

    def post_process(self, im):
        return im

    def process(self, im):
        im = self.pre_process(im)
        im = self.post_process(im)
        return im

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        try:
            default_storage.delete(self.sample_filename())
        except:
            pass
        models.Model.save(self, *args, **kwargs)
        self.create_sample()
        for size in self.photo_sizes.all():
            size.clear_cache()
        # try to clear all related subclasses of ImageModel
        for prop in [prop for prop in dir(self) if prop[-8:] == '_related']:
            for obj in getattr(self, prop).all():
                obj.clear_cache()
                obj.pre_cache()

    def delete(self):
        try:
            default_storage.delete(self.sample_filename())
        except:
            pass
        models.Model.delete(self)


class PhotoEffect(BaseEffect):

    """ A pre-defined effect to apply to photos """
    transpose_method = models.CharField(_('rotate or flip'),
                                        max_length=15,
                                        blank=True,
                                        choices=IMAGE_TRANSPOSE_CHOICES)
    color = models.FloatField(_('color'),
                              default=1.0,
                              help_text=_('A factor of 0.0 gives a black and white image, a factor of 1.0 gives the '
                                          'original image.'))
    brightness = models.FloatField(_('brightness'),
                                   default=1.0,
                                   help_text=_('A factor of 0.0 gives a black image, a factor of 1.0 gives the '
                                               'original image.'))
    contrast = models.FloatField(_('contrast'),
                                 default=1.0,
                                 help_text=_('A factor of 0.0 gives a solid grey image, a factor of 1.0 gives the '
                                             'original image.'))
    sharpness = models.FloatField(_('sharpness'),
                                  default=1.0,
                                  help_text=_('A factor of 0.0 gives a blurred image, a factor of 1.0 gives the '
                                              'original image.'))
    filters = models.CharField(_('filters'),
                               max_length=200,
                               blank=True,
                               help_text=_(IMAGE_FILTERS_HELP_TEXT))
    reflection_size = models.FloatField(_('size'),
                                        default=0,
                                        help_text=_('The height of the reflection as a percentage of the orignal '
                                                    'image. A factor of 0.0 adds no reflection, a factor of 1.0 adds a'
                                                    ' reflection equal to the height of the orignal image.'))
    reflection_strength = models.FloatField(_('strength'),
                                            default=0.6,
                                            help_text=_('The initial opacity of the reflection gradient.'))
    background_color = models.CharField(_('color'),
                                        max_length=7,
                                        default="#FFFFFF",
                                        help_text=_('The background color of the reflection gradient. Set this to '
                                                    'match the background color of your page.'))

    class Meta:
        verbose_name = _("photo effect")
        verbose_name_plural = _("photo effects")

    def pre_process(self, im):
        if self.transpose_method != '':
            method = getattr(Image, self.transpose_method)
            im = im.transpose(method)
        if im.mode != 'RGB' and im.mode != 'RGBA':
            return im
        for name in ['Color', 'Brightness', 'Contrast', 'Sharpness']:
            factor = getattr(self, name.lower())
            if factor != 1.0:
                im = getattr(ImageEnhance, name)(im).enhance(factor)
        for name in self.filters.split('->'):
            image_filter = getattr(ImageFilter, name.upper(), None)
            if image_filter is not None:
                try:
                    im = im.filter(image_filter)
                except ValueError:
                    pass
        return im

    def post_process(self, im):
        if self.reflection_size != 0.0:
            im = add_reflection(im, bgcolor=self.background_color,
                                amount=self.reflection_size, opacity=self.reflection_strength)
        return im


class Watermark(BaseEffect):
    image = models.ImageField(_('image'),
                              upload_to=PHOTOLOGUE_DIR + "/watermarks")
    style = models.CharField(_('style'),
                             max_length=5,
                             choices=WATERMARK_STYLE_CHOICES,
                             default='scale')
    opacity = models.FloatField(_('opacity'),
                                default=1,
                                help_text=_("The opacity of the overlay."))

    class Meta:
        verbose_name = _('watermark')
        verbose_name_plural = _('watermarks')

    def delete(self):
        assert self._get_pk_val() is not None, "%s object can't be deleted because its %s attribute is set to None." \
            % (self._meta.object_name, self._meta.pk.attname)
        super(Watermark, self).delete()
        self.image.storage.delete(self.image.name)

    def post_process(self, im):
        mark = Image.open(self.image.storage.open(self.image.name))
        return apply_watermark(im, mark, self.style, self.opacity)


class PhotoSize(models.Model):

    """About the Photosize name: it's used to create get_PHOTOSIZE_url() methods,
    so the name has to follow the same restrictions as any Python method name,
    e.g. no spaces or non-ascii characters."""

    name = models.CharField(_('name'),
                            max_length=40,
                            unique=True,
                            help_text=_(
                                'Photo size name should contain only letters, numbers and underscores. Examples: '
                                '"thumbnail", "display", "small", "main_page_widget".'),
                            validators=[RegexValidator(regex='^[a-z0-9_]+$',
                                                       message='Use only plain lowercase letters (ASCII), numbers and '
                                                       'underscores.'
                                                       )]
                            )
    width = models.PositiveIntegerField(_('width'),
                                        default=0,
                                        help_text=_(
                                            'If width is set to "0" the image will be scaled to the supplied height.'))
    height = models.PositiveIntegerField(_('height'),
                                         default=0,
                                         help_text=_(
        'If height is set to "0" the image will be scaled to the supplied width'))
    quality = models.PositiveIntegerField(_('quality'),
                                          choices=JPEG_QUALITY_CHOICES,
                                          default=70,
                                          help_text=_('JPEG image quality.'))
    upscale = models.BooleanField(_('upscale images?'),
                                  default=False,
                                  help_text=_('If selected the image will be scaled up if necessary to fit the '
                                              'supplied dimensions. Cropped sizes will be upscaled regardless of this '
                                              'setting.')
                                  )
    crop = models.BooleanField(_('crop to fit?'),
                               default=False,
                               help_text=_('If selected the image will be scaled and cropped to fit the supplied '
                                           'dimensions.'))
    pre_cache = models.BooleanField(_('pre-cache?'),
                                    default=False,
                                    help_text=_('If selected this photo size will be pre-cached as photos are added.'))
    increment_count = models.BooleanField(_('increment view count?'),
                                          default=False,
                                          help_text=_('If selected the image\'s "view_count" will be incremented when '
                                                      'this photo size is displayed.'))
    effect = models.ForeignKey('photologue.PhotoEffect',
                               null=True,
                               blank=True,
                               related_name='photo_sizes',
                               verbose_name=_('photo effect'),
                               on_delete=models.CASCADE)
    watermark = models.ForeignKey('photologue.Watermark',
                                  null=True,
                                  blank=True,
                                  related_name='photo_sizes',
                                  verbose_name=_('watermark image'),
                                  on_delete=models.CASCADE)

    class Meta:
        ordering = ['width', 'height']
        verbose_name = _('photo size')
        verbose_name_plural = _('photo sizes')

    def __str__(self):
        return self.name

    def clear_cache(self):
        for cls in ImageModel.__subclasses__():
            for obj in cls.objects.all():
                obj.remove_size(self)
                if self.pre_cache:
                    obj.create_size(self)
        PhotoSizeCache().reset()

    def clean(self):
        if self.crop is True:
            if self.width == 0 or self.height == 0:
                raise ValidationError(
                    _("Can only crop photos if both width and height dimensions are set."))

    def save(self, *args, **kwargs):
        super(PhotoSize, self).save(*args, **kwargs)
        PhotoSizeCache().reset()
        self.clear_cache()

    def delete(self):
        assert self._get_pk_val() is not None, "%s object can't be deleted because its %s attribute is set to None." \
            % (self._meta.object_name, self._meta.pk.attname)
        self.clear_cache()
        super(PhotoSize, self).delete()

    def _get_size(self):
        return (self.width, self.height)

    def _set_size(self, value):
        self.width, self.height = value
    size = property(_get_size, _set_size)


class PhotoSizeCache(object):
    __state = {"sizes": {}}

    def __init__(self):
        self.__dict__ = self.__state
        if not len(self.sizes):
            sizes = PhotoSize.objects.all()
            for size in sizes:
                self.sizes[size.name] = size

    def reset(self):
        global size_method_map
        size_method_map = {}
        self.sizes = {}


def init_size_method_map():
    global size_method_map
    for size in PhotoSizeCache().sizes.keys():
        size_method_map['get_%s_size' % size] = \
            {'base_name': '_get_SIZE_size', 'size': size}
        size_method_map['get_%s_photosize' % size] = \
            {'base_name': '_get_SIZE_photosize', 'size': size}
        size_method_map['get_%s_url' % size] = \
            {'base_name': '_get_SIZE_url', 'size': size}
        size_method_map['get_%s_filename' % size] = \
            {'base_name': '_get_SIZE_filename', 'size': size}


# def add_default_site(instance, created, **kwargs):
#     """
#     Called via Django's signals when an instance is created.
#     In case PHOTOLOGUE_MULTISITE is False, the current site (i.e.
#     ``settings.SITE_ID``) will always be added to the site relations if none are
#     present.
#     """
#     if not created:
#         return
#     if getattr(settings, 'PHOTOLOGUE_MULTISITE', False):
#         return
#     if instance.sites.exists():
#         return
#     instance.sites.add(Site.objects.get_current())
# post_save.connect(add_default_site, sender=Gallery)
# post_save.connect(add_default_site, sender=Photo)



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
        super().save()

    def save_when_name_not_inited(self):
        # TODO: Fix this strange saving.
        super().save()
        self.name = Path(self.image.name).name
        super().save()

    def get_absolute_url(self):
        return reverse('photologue:database_photo_detail', args=[self.slug])



class EventPhoto(ImageModel):

    # TODO: Make slug unique?
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
    similarity = models.FloatField(null=True,
                                   blank=True)

    def get_absolute_url(self):
        return reverse('photologue:event_photo_detail', args=[self.event.slug, self.pk])

    def __str__(self):
        return f'{self.database_photo.name} from {self.event}'

    def save(self):
        full_path_to_original_photo = os.path.join(settings.MEDIA_ROOT, self.database_photo.image.name)
        self.copy_photo_and_assign_image_field(full_path_to_original_photo)
        super().save()
