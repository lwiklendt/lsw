from concurrent import futures


def parexec(exec_func, nblocks, nthreads=None, *args, **kwargs):
    if nthreads is None:
        nthreads = nblocks
    with futures.ThreadPoolExecutor(nthreads) as executor:
        fs = futures.wait([executor.submit(exec_func, i, *args, **kwargs) for i in range(nblocks)])
    return [f.result() for f in fs.done]


def parexec_proc(exec_func, nblocks, nprocs=None, *args, **kwargs):
    if nprocs is None:
        nprocs = nblocks
    with futures.ProcessPoolExecutor(nprocs) as executor:
        fs = futures.wait([executor.submit(exec_func, i, *args, **kwargs) for i in range(nblocks)])
    return [f.result() for f in fs.done]
