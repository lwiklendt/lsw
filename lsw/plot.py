import numpy as np


def extent(x, y):
    dx = np.diff(x[:2])[0]
    dy = np.diff(y[:2])[0]
    return np.array([x[0], x[-1], y[0], y[-1]]) + 0.5 * np.array([-dx, dx, -dy, dy])


def edges_to_centers(edges, log=False):
    if log:
        edges = np.log2(edges)
    centers = edges[1:] - 0.5 * (edges[1] - edges[0])
    if log:
        centers = 2 ** centers
    return centers


def centers_to_edges(centers, log=False):
    if log:
        centers = np.log2(centers)
    dx = centers[1] - centers[0]
    edges = np.r_[centers, centers[-1] + dx] - 0.5 * dx
    if log:
        edges = 2 ** edges
    return edges


def edge_meshgrid(centers_x, centers_y, logx=False, logy=False):
    return np.meshgrid(centers_to_edges(centers_x, logx), centers_to_edges(centers_y, logy))


def fig2rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape((h, w, 3))


def fig2bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape((h, w, 3))[:, :, ::-1]
