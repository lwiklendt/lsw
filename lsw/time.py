import datetime


# return the time difference from 0:00:00
def time_to_timedelta(t):
    if type(t) is datetime.time:
        return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    else:
        return None


# decorator for timing and printing elapsed time of a function call
def timeme(func):
    def wrapper(*args, **kwargs):
        tstart = datetime.datetime.now()
        result = func(*args, **kwargs)
        print(f'\033[1m{func.__name__}\033[0m timing: {datetime.datetime.now() - tstart}')
        return result
    return wrapper


def seconds_to_hms_str(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    hms_str = ''
    if h > 0:
        hms_str += ' {:g}h'.format(h)
    if m > 0:
        hms_str += ' {:g}m'.format(m)
    if s > 0:
        hms_str += ' {:g}s'.format(s)
    return hms_str.strip()
