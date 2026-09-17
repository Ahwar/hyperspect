import numpy as np
from PIL import Image

def X2Cube(img, cellSize=4):
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)

img = np.array(Image.open('bin/raw/data_train/data_train/VIS/3.png'))
cube = X2Cube(img)

bands = [0, 1, 2]
# bands = [5, 8, 13]
pseudo_rgb = cube[:, :, bands].astype(np.float32)

for c in range(3):
    ch = pseudo_rgb[:, :, c]
    vmin, vmax = ch.min(), ch.max()
    if vmax > vmin:
        pseudo_rgb[:, :, c] = ((ch - vmin) / (vmax - vmin) * 255)
pseudo_rgb = pseudo_rgb.astype(np.uint8)

Image.fromarray(pseudo_rgb).save('pseudo_rgb.jpg')
