import os
import pickle
from pathlib import Path

from cbir import SESSIONS
from cbir import exceptions


class Session:
    legacy_extensions = ['pkl']
    extension = legacy_extensions[-1]

    def __init__(self, database, query, query_saved=False):
        self.database = database
        self.query = query
        self.query_saved = query_saved
        self.basket = []

    def add(self, image):
        """
        :param image: full path to image from database
        """
        self.basket.append(image)

    def add_list(self, images):
        self.basket.extend(images)

    def save(self, tag):
        destination = Path(SESSIONS) / self.database / tag

        # QUESTION: Problems if concurrent access?
        if not os.path.exists(Path(SESSIONS) / self.database):
            os.mkdir(Path(SESSIONS) / self.database)

        if os.path.exists(f'{destination}.{Session.extension}'):
            raise exceptions.DuplicateSessionTag()

        self._save(f'{destination}.{Session.extension}')

    @classmethod
    def load(cls, database, tag):
        destination = Path(SESSIONS) / database / tag
        if not os.path.exists(f'{destination}.{Session.extension}'):
            raise exceptions.CBIRException(f'Path {destination}.{Session.extension} does not exist')

        return cls._load(f'{destination}.{Session.extension}')

    def _save(self, path):
        with open(path, 'wb') as fout:
            pickle.dump({'basket': self.basket,
                         'query': self.query,
                         'query_saved': self.query_saved,
                         'database': self.database},
                        fout, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _load(cls, path):
        session = cls.__new__(cls)
        with open(path, 'rb') as fin:
            tmp = pickle.load(fin)
            session.basket = tmp['basket']
            session.query = tmp['query']
            session.query_saved = tmp['query_saved']
            session.database = tmp['database']
        return session
