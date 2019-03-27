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
                pathname = os.path.split(cache_filename)[0]
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                with open(cache_filename, 'wb') as f:
                    pickle.dump(result, f, protocol=2)
            else:
                with open(cache_filename, 'rb') as f:
                    result = pickle.load(f)
            return result
        return wrapper
    return decorator


def isnewer(src, dst):
    if os.path.exists(dst):
        return os.path.getmtime(src) > os.path.getmtime(dst)
    else:
        return True


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
