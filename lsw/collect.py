import collections


# https://stackoverflow.com/a/2158532/142712
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def permutation_swaps(n):
    """
    Heap's algorithm for generating permutation swaps.
    :param n: number of elements to permute
    :return: generator yielding a pair of indexes for which pair of elements should be swapped
    """
    c = [0] * n
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                yield (0, i)
            else:
                yield (c[i], i)
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
