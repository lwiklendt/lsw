from concurrent import futures


def parexec(exec_func, nblocks, nworkers=None, *args, **kwargs):
    if nworkers is None:
        nworkers = nblocks
    with futures.ThreadPoolExecutor(nworkers) as executor:
        fs = futures.wait([executor.submit(exec_func, i, *args, **kwargs) for i in range(nblocks)])
    return [f.result() for f in fs.done]


def parexec_proc(exec_func, nblocks, nworkers=None, *args, **kwargs):
    if nworkers is None:
        nworkers = nblocks
    with futures.ProcessPoolExecutor(nworkers) as executor:
        fs = futures.wait([executor.submit(exec_func, i, *args, **kwargs) for i in range(nblocks)])
    return [f.result() for f in fs.done]
