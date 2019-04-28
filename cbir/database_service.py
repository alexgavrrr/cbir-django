from pathlib import Path
import threading
import sqlite3

import peewee
from .database_schema import (Photo,
                              Word,
                              WordPhoto)

MODELS_LOCK = threading.Lock()


def get_db(path):
    db = peewee.SqliteDatabase(str(Path(path) / 'index.db'))
    return db


def inited_properly(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word, WordPhoto]):
            return Photo.table_exists() and Word.table_exists() and WordPhoto.table_exists()


def create_empty(db):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            db.create_tables([Photo, Word, WordPhoto])
    return db


def add_photos(
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
            with db.atomic():
                Photo.insert_many(photos).execute()


def update_bows(db, photos):
    """
    :param db:
    :param photos: one photo: {
                        'name': ...,
                        'bow': ...,}
    """
    # TODO: Validate here that keys in objects are fine?
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            with db.atomic():
                photos = [Photo(name=photo['name'], bow=photo['bow'])
                          for photo
                          in photos]
                Photo.bulk_update(photos, fields=[Photo.bow])


def get_photos_descriptors_by_names_iterator(
        db,
        names):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            # TODO: Order of photos is important or not?
            return ({
                'descriptor': Photo
                    .select(Photo.descriptor)
                    .where(Photo.name == name)[0].descriptor}
                for name
                in names)


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
                     .where((Photo.to_index == True) & (Photo.bow == None)))
            return query.iterator()


def get_photos_by_words_iterator(
        db,
        words):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            query = (Word
                     .select(Word.photos)
                     .where(Word.word in words))
            return query.dicts().iterator()


def is_image_indexed(db, name):
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word]):
            return (Photo
                    .select()
                    .where(Photo.name == name).count()
                    == 1)


def count_indexed(db):
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
            with db.atomic():
                WordPhoto.insert_many(relations).execute()


def clean_word_table(db):
    raise NotImplemented


def insert_or_replace_word(db, words_photos):
    """
    :param db:
    :param words_photos: one word = {
                'word': ...,
                'photos': ...,}
    :return:
    """
    with MODELS_LOCK:
        with db.bind_ctx([Photo, Word, WordPhoto]):
            with db.atomic():
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
        with db.bind_ctx([Photo]):
            photo = (Photo
                     .get(name=name))
            return {'bow': photo.bow}
