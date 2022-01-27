import numpy as np


def create_patches_overlap(im, m, A=None):
    r, c = im.shape
    patches = []
    patches_a = []
    for i in range(r):
        for j in range(c):
            if i + m <= r and j + m <= c:
                patches.append((im[i:i+m, j:j+m]).reshape(m*m, 1))
                if A is not None:
                    patches_a.append((A[i:i+m, j:j+m]).reshape(m*m, 1))
    result_y = np.concatenate(patches, axis=1)
    if A is not None:
        return result_y, np.concatenate(patches_a, axis=1)
    else:
        return result_y, (result_y != 0).astype(float)


def patch_average(patch, m, r, c):
    im = np.zeros((r, c))
    avg = np.zeros((r, c))
    cpt = 0
    for i in range(r):
        for j in range(c):
            if i+m <= r and j+m <= c:
                im[i:i+m, j:j+m] += patch[:, cpt].reshape(m, m)
                avg[i:i+m, j:j+m] += np.ones((m, m))
                cpt += 1
    return im / avg
