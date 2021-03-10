import functools
import os
import pickle


# decorator for pickle-caching the result of a function
def pickle_cache(cache_filename, compare_filename_time=None, overwrite=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            exists = os.path.exists(cache_filename)
            needs_redo = overwrite
            if exists and compare_filename_time is not None:
                needs_redo |= os.path.getmtime(cache_filename) < os.path.getmtime(compare_filename_time)
            if not exists or needs_redo:
                result = func(*args, **kwargs)
                pkl_save(result, cache_filename)
            else:
                result = pkl_load(cache_filename)
            return result
        return wrapper
    return decorator


def pkl_save(obj, filename):
    pathname = os.path.split(filename)[0]
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result


def isnewer(src, dst):
    if os.path.exists(dst):
        return os.path.getmtime(src) > os.path.getmtime(dst)
    else:
        return True


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
