import numpy as np
from numba import njit


@njit
def permutation_order(w):
    """
    Computes the best ordering of a bipartite graph with edge weights given by the 2D array w. The loss for an ordering
    is given by the sum of crossing weights, where the crossing weight is given by the product of the weights of the
    pair of edges involved in that crossing. The method is a brute-force search through all permutations of the second
    layer vertices using Heap's algorithm https://en.wikipedia.org/wiki/Heap%27s_algorithm.
    :param w: Edge weights, where w[i,j] is the weight of edge from vertex i in the first layer to vertex j in the
    second. This array will get overwritten/scrambled during the procedure. Copy the input to keep the original weights
    before passing to this function.
    :return: An ordering of the second layer which results in lowest crossing loss.
    """

    m, n = w.shape
    order = np.arange(n)

    # compute initial loss
    best_loss = 0
    for mi in range(m - 1):
        for ni in range(n - 1):
            best_loss += w[mi, ni] * np.sum(w[mi + 1:, ni + 1:])
    best_order = order.copy()

    # Heap's permutation algorithm produces pairs of indexes to swap for the next permutation
    c = np.zeros_like(order)
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                a, b = 0, i
            else:
                a, b = c[i], i

            # --- we have indexes a and b to swap, so swap in w and order, and compute the loss for this ordering

            # swap w[:, a] w[:, b]
            x = w[:, a].copy()
            w[:, a] = w[:, b]
            w[:, b] = x

            # swap order[a] order[b]
            o = order[a]
            order[a] = order[b]
            order[b] = o

            # compute loss for this permutation's ordering
            loss = 0
            for mi in range(m - 1):
                for ni in range(n - 1):
                    loss_mn = 0
                    for mj in range(mi + 1, m):
                        for nj in range(ni + 1, n):
                            loss_mn += w[mj, nj]
                    loss += w[mi, ni] * loss_mn

            # keep if best loss so far
            if loss < best_loss:
                best_loss = loss
                best_order = order.copy()

            # --- done loss computation for this permutation

            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1

    return best_order
