from concurrent import futures
import multiprocessing


def parexec(exec_func, nblocks, nthreads=None):
    if nthreads is None:
        nthreads = multiprocessing.cpu_count()
    with futures.ThreadPoolExecutor(nthreads) as executor:
        futures.wait([executor.submit(exec_func, i) for i in range(nblocks)])
