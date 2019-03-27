from concurrent import futures


def parexec(exec_func, nblocks, nthreads=None):
    if nthreads is None:
        nthreads = nblocks
    with futures.ThreadPoolExecutor(nthreads) as executor:
        futures.wait([executor.submit(exec_func, i) for i in range(nblocks)])
