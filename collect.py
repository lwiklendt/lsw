import collections
from typing import List, Tuple, Iterator


def strings(a: List[Tuple]) -> Iterator[List]:
    """
    Generates all unique lists of length len(a) such that for each generatred item x, x[i] in a[i].
    Example: strings([('a', 'b'), (1, 2, 3)]) yields ['a', 1], ['b', 1], ['a', 2], ['b', 2], ['a', 3], ['b', 3].
    :param a: list containing the alphabet at each index from which generated lists obtain their values.
    """
    counts = [len(l) for l in a]
    n = len(a)
    s = [0, ] * n  # holds the currently incrementing state as indexes into the tuples in a
    while True:
        yield [a[i][s[i]] for i in range(n)]
        i = 0
        s[i] += 1
        while s[i] == counts[i]:
            s[i] = 0
            i += 1
            if i == n:
                return
            s[i] += 1


# https://stackoverflow.com/a/2158532/142712
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def permutation_swaps(n: int) -> Iterator[Tuple[int, int]]:
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
                yield 0, i
            else:
                yield c[i], i
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1


def sub(xs, ys):
    """
    Computes xs - ys, such that elements in xs that occur in ys are removed.
    @param xs: list
    @param ys: list
    @return: xs - ys
    """
    return [x for x in xs if x not in ys]


def list_of_dicts_to_dict_of_lists(ld):
    """
    Thanks to Andrew Floren from https://stackoverflow.com/a/33046935/142712
    :param ld: list of dicts
    :return: dict of lists
    """
    return {k: [d[k] for d in ld] for k in ld[0]}


def dict_of_lists_to_list_of_dicts(dl):
    """
    Thanks to Andrew Floren from https://stackoverflow.com/a/33046935/142712
    :param dl: dict of lists
    :return: list of dicts
    """
    return [dict(zip(dl, t)) for t in zip(*dl.values())]
