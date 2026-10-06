import scipy.linalg
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    data = np.load(filename)
    data = data - np.mean(data, axis=0)
    return data


def get_covariance(dataset):
    orig = np.array(dataset)
    trans = np.transpose(orig)

    # dots
    dotp = np.dot(trans, orig)
    dotp = np.divide(dotp, len(orig) - 1)

    return dotp


def get_eig(S, m):
    vals, vectors = scipy.linalg.eigh(S, subset_by_index=[len(S) - m, len(S[0]) - 1])

    # rearrange and reformat value output
    vals = sorted(vals, reverse=True)
    vals = np.diag(vals)

    # rearrange vector output
    for i in range(0, len(vectors)):
        vectors[i] = sorted(vectors[i], reverse=True)

    return vals, vectors


def get_eig_prop(S, prop):
    # get vals and vectors
    bound = np.multiply(prop, np.trace(S))
    vals, vectors = scipy.linalg.eigh(S, subset_by_value=[bound, np.inf])

    # rearrange and reformat value output
    vals = sorted(vals, reverse=True)
    vals = np.diag(vals)

    # rearrange vector output
    for i in range(0, len(vectors)):
        vectors[i] = sorted(vectors[i], reverse=True)

    return vals, vectors


def project_image(image, U):
    for i in range(len(U)):
        trans = np.transpose(U)
        # weight
        alpha = np.dot(trans, image)
        # sum
        proj = np.dot(alpha, trans)
    return proj


def display_image(orig, proj_reshape):
    # reshape
    orig_reshape = np.reshape(orig, [32, 32])
    orig_reshape = np.transpose(orig_reshape)
    proj_reshape = np.reshape(proj_reshape, [32, 32])
    proj_reshape = np.transpose(proj_reshape)

    # setup
    fig, (p1, p2) = plt.subplots(1, 2)

    # img displays
    p1.set_title("Original")
    p2.set_title("Projection")
    display = p1.imshow(orig_reshape, aspect='equal')
    display2 = p2.imshow(proj_reshape, aspect='equal')

    # colorbars
    fig.colorbar(display, ax=p1)
    fig.colorbar(display2, ax=p2)

    plt.show()


# TESTER GRAVEYARD

def main():
    x = load_and_center_dataset('YaleB_32x32.npy')
    y = get_covariance(x)
    lammy, v = get_eig(y, 2)
    lammy2, v2 = get_eig_prop(y, 0.07)
    projection = project_image(x[0], v2)
    display_image(x[0], projection)


if __name__ == '__main__':
    main()
