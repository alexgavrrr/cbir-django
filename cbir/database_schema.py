import logging
from pathlib import Path

from peewee import *
from peewee import Model

from cbir import DATABASES

PATH_TO_FALLBACK_DATABASE = str(Path(DATABASES) / 'fallback.db')


class CustomProxy(DatabaseProxy):
    def __getattr__(self, attr):
        if self.obj is None:
            self.initialize_proxy()
        return super().__getattr__(attr)

    def initialize_proxy(self):
        logging.getLogger().debug("Access to uninitialized DB proxy requested. Try initialization ...")
        lazy_loaded_db = SqliteDatabase(PATH_TO_FALLBACK_DATABASE)
        self.initialize(lazy_loaded_db)


global_database_proxy = CustomProxy()


class BaseModel(Model):
    class Meta:
        database = global_database_proxy


class Photo(BaseModel):
    # pk = PrimaryKeyField()
    name = CharField(max_length=250,
                     primary_key=True,
                     unique=True,
                     index=True)
    descriptor = BlobField()
    to_index = BooleanField()
    for_training = BooleanField()


class PhotoBow(BaseModel):
    # pk = PrimaryKeyField()
    name = CharField(max_length=250,
                     primary_key=True,
                     unique=True,
                     index=True)
    bow = BlobField(null=False)



class Word(BaseModel):
    """
    Class for retrieving by word
    """
    word = PrimaryKeyField()
    photos = BlobField(null=False)


class WordPhoto(BaseModel):
    """
    Class for fast adding word - photo relation
    """
    word = IntegerField()

    # photo = ForeignKeyField(Photo)  Foreigh key slower?
    photo = CharField(max_length=250)
