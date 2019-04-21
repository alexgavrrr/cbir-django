from peewee import *
from peewee import Model

database_proxy = DatabaseProxy()


class Photo(Model):
    # pk = PrimaryKeyField()
    name = CharField(max_length=250,
                     primary_key=True,
                     unique=True,
                     index=True)
    descriptor = BlobField()
    bow = BlobField(null=True)

    to_index = BooleanField()
    for_training = BooleanField()

    class Meta:
        database = database_proxy


class Word(Model):
    """
    Class for retrieving by word
    """
    word = PrimaryKeyField()
    photos = BlobField(null=False)

    class Meta:
        database = database_proxy


class WordPhoto(Model):
    """
    Class for fast adding word - photo relation
    """
    word = IntegerField(index=True)
    photo = ForeignKeyField(Photo)
