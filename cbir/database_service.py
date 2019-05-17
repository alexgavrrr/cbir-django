import threading
from pathlib import Path

import peewee
from playhouse.migrate import (migrate,
                               SqliteMigrator)

from .database_schema import (Photo,
                              PhotoBow,
                              Word,
                              WordPhoto)

MODELS_LOCK = threading.Lock()


def clean_word_photo_relations_table(db):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
            if db.table_exists('wordphoto'):
                WordPhoto.drop_table()

            migrator = SqliteMigrator(db)
            try:
                migrate(migrator.drop_index('wordphoto', 'word'))
            except peewee.OperationalError:
                pass

            WordPhoto.create_table()


def clean_bow(db):
    with MODELS_LOCK:
        with db.bind_ctx([PhotoBow]):
            if db.table_exists('photobow'):
                PhotoBow.drop_table()
            PhotoBow.create_table()


def clean_word(db):
    with MODELS_LOCK:
        with db.bind_ctx([Word]):
            if db.table_exists('word'):
                Word.drop_table()
            Word.create_table()


def create_if_needed_word_photo_relations_table(db):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
            if not db.table_exists('wordphoto'):
                WordPhoto.create_table()


def delete_index_if_needed_in_word_photo_relations(db):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
                migrator = SqliteMigrator(db)
                try:
                    migrate(migrator.drop_index('wordphoto', 'wordphoto_word'), )
                except peewee.OperationalError as exc:
                    pass


def sort_word_photo_relations_table(db):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
            migrator = SqliteMigrator(db)
            migrate(migrator.add_index('wordphoto', ('word',), False), )


def get_db(path):
    db = peewee.SqliteDatabase(str(Path(path) / 'index.db'))
    return db


def inited_properly(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, PhotoBow, Word]):
            return Photo.table_exists() and PhotoBow.table_exists() and Word.table_exists()


def create_empty(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, PhotoBow, Word]):
            db.create_tables([Photo, PhotoBow, Word])
    return db


def add_photos_descriptors(
        db,
        photos):
    """
    :param db:
    :param photos: one photo: {
                        'name': ...,
                        'descriptor': ...,
                        'to_index': ...,
                        'for_training': ...,}
    """
    # TODO: Validate here that keys in objects are fine?
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            # with db.atomic():
                Photo.insert_many(photos).execute()


def write_bows(db, photos_bows):
    """
    :param db:
    :param photos_bows: one photo: {
                        'name': ...,
                        'bow': ...,}
    """
    # TODO: Validate here that keys in objects are fine?
    with MODELS_LOCK:
        with db.bind_ctx([Photo, PhotoBow, Word]):
            # with db.atomic():
                (PhotoBow
                 .insert_many(photos_bows)
                 .on_conflict('replace')
                 .execute())


def get_photos_descriptors_by_names_iterator(
        db,
        names):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            # return ({
            #     'descriptor': Photo
            #         .select(Photo.descriptor)
            #         .where(Photo.name == name)[0].descriptor}
            #     for name
            #     in names)

            # TODO: Order of photos is important or not?
            query = (Photo
                     .select(Photo.name, Photo.descriptor)
                     .where(Photo.name << names))
            return query.dicts().iterator()


def get_photos_descriptors_for_training_iterator(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            query = (Photo
                     .select(Photo.descriptor)
                     .where(Photo.for_training == True))
            return query.dicts().iterator()


def get_photos_descriptors_needed_to_add_to_index_iterator(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            query = (Photo
                     .select(Photo.name, Photo.descriptor)
                     # .where((Photo.to_index == True) & (Photo.indexed == False))
                     .where((Photo.to_index == True))
                     )

            return query.iterator()


def get_photos_by_words_iterator(
        db,
        words):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            query = (Word
                     .select(Word.photos)
                     .where(Word.word << words))
            return query.dicts().iterator()


def is_image_descriptor_computed(db, name):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            return (Photo
                    .select()
                    .where(Photo.name == name).count()
                    == 1)


def count_for_indexing(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            return (Photo
                    .select()
                    .where(Photo.to_index == True)
                    .count())


def count_for_training(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            return (Photo
                    .select()
                    .where(Photo.for_training == True)
                    .count())


def add_word_photo_relations(db, relations):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
            # with db.atomic():
                WordPhoto.insert_many(relations).execute()


def insert_or_replace_words(db, words_photos):
    """
    :param db:
    :param words_photos: one word = {
                'word': ...,
                'photos': ...,}
    :return:
    """
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word, WordPhoto]):
            # with db.atomic():
                (Word
                 .insert_many(words_photos)
                 .on_conflict('replace')
                 .execute())


def get_word_photo_relations_sorted(db: peewee.Database):
    with MODELS_LOCK:
        with db.bind_ctx([WordPhoto], bind_refs=False, bind_backrefs=False):
            query = (WordPhoto
                     .select(WordPhoto.word, WordPhoto.photo)
                     .order_by(WordPhoto.word))
            return query.dicts().iterator()


def get_bow(db: peewee.Database, name):
    with MODELS_LOCK:
        with db.bind_ctx([PhotoBow]):
            photo_bow = (PhotoBow
                         .get(name=name))
            return {'bow': photo_bow.bow}
