# CS540 HW3 Spring 2023

## Required Packages
- NumPy `import numpy as np`
- SciPy (version >= 1.5.0) `from scipy.linalg import eigh`
- matplotlib `import matplotlib.pyplot as plt`

## Dataset
This description is copied directly from the assignment specification:
> You will be using part of the Yale face dataset (processed). The dataset is saved in the ’YaleB_32x32.npy’
file. The ’.npy’ file format is used to store numpy arrays. We will test your code only using this provided
dataset.
The dataset contains 2414 sample images, each of size 32 × 32. We will use n to refer to the number of
images (so n = 2414) and d to refer to the number of features for each sample image (so d = 1024 = 32×32).
Note, we’ll use xi to refer to the ith sample image which is a d-dimensional feature vector.

## Implemented functions
1. `load_and_center_dataset(filename)`: loads and centers provided dataset about the origin. Returns as numpy array of floats
2. `get_covariance(dataset)`: calculates and returns covariance matrix of dataset as numpy matrix
3. `get_eig(S,m)`: returns a diagonal matrix with largest eigenvalues in descending order as a numpy array and another matrix with corresponding eigenvectors as columns in another numpy array
4. `get_eig_prop(S,prop)`: returns **all** eigenvalues and corresponding vectors in similar format
5. `project_image(image,U)`: projects image into *m*-dimensional subspace then back into *d X 1* and returns
6. `display_image(orig,proj)`: displays visual representation of original and projected image side by side with `matplotlib`
