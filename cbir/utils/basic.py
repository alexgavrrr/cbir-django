import functools
import os
import time
import rapidjson
import shutil
from pathlib import Path


def timeit_my(func):
    @functools.wraps(func)
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = round(time.time() - start, 3)
        return elapsed, result
    return timed


def generate_timestamp():
    """
    The Unix time, rounded to the nearest second.
    See https://en.wikipedia.org/wiki/Unix_time

    :return: (`str`) the Unix time
    """
    return str(round(time.time()))


def serialize(data):
    # TODO: Change serialization or ensure that this is fine or get rid of it.
    return rapidjson.dumps(data, skipkeys=False, ensure_ascii=False, sort_keys=True)


def deserialize(data):
    return rapidjson.loads(data)


def copy_files(source, destination, predicate, prefix_for_conflicted_files):
    """
    :param source:
    :param destination:
    :param predicate:
    :param prefix_for_conflicted_files: if None or '' or False then skip.
    """
    if not os.path.isdir(source):
        message = f'Source {source} must be valid directory'
        raise ValueError(message)

    if not os.path.isdir(destination):
        message = f'Destination {destination} must be valid directory'
        raise ValueError(message)

    original_filenames_in_destination = set(os.listdir(destination))
    for filename in os.listdir(source):
        if (predicate(filename)
                and (filename not in original_filenames_in_destination or prefix_for_conflicted_files)):
            prefix = (''
                      if filename not in original_filenames_in_destination
                      else prefix_for_conflicted_files)
            shutil.copyfile(Path(source) / filename,
                            Path(destination) / (prefix + filename))


def is_image(filename):
    # TODO: Remove
    if filename[filename.rfind('.'):] in ['jpg', 'png']:
        raise ValueError('Hey look here bug')

    return filename[filename.rfind('.'):] in ['.jpg', '.png']
