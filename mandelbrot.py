import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from tqdm import tqdm

cmap = matplotlib.cm.nipy_spectral
cmap.set_bad('black')

RES_STEP = 0.0001
X_MIN = -0.7
X_MAX = -0.5
Y_MIN = 0.4
Y_MAX = 0.6
MAX_N = 300
MAGNITUDE_THRESHOLD = 3.

DPI = 400


def f(z, c):
    return z**2 + c


def iterate_convergence(plane):
    z = np.zeros(plane.shape)
    iteration_plane = np.zeros(plane.shape)
    for _ in tqdm(range(MAX_N)):
        z = f(z, plane)
        iteration_plane[np.abs(z) < MAGNITUDE_THRESHOLD] += 1
        z[np.abs(z) > MAGNITUDE_THRESHOLD] = MAGNITUDE_THRESHOLD
    iteration_plane[iteration_plane == MAX_N] = np.nan
    return iteration_plane


def plot_fig(output):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(output, cmap=cmap, interpolation='nearest')
    ax.set_axis_off()
    plt.tight_layout()
    file_path = f'outputs/x_min-{X_MIN}--x_max-{X_MAX}--y_min-{Y_MIN}--y_max-{Y_MAX}--res_step-{RES_STEP}-mandelbrot.png'
    plt.savefig(file_path, dpi=DPI)
    print(f'Output saved to {file_path} (options: DPI: {DPI})')


def run_simulation():
    xstep = np.arange(X_MIN, X_MAX, RES_STEP)
    ystep = np.arange(Y_MIN, Y_MAX, RES_STEP)
    plane = xstep[:, None] + 1j*ystep
    result = iterate_convergence(plane).T
    plot_fig(result)


if __name__ == '__main__':
    run_simulation()
