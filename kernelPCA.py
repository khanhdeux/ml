# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from lib import get_dummy_data

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation
    
    Parameters
    ----------
    X: {Numpy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Tuning parameter of the RBF Kernel
    
    n_components: int
        Number of principal components to return

    Returns
    ----------
    X_pc {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset        
    """
    
    # Calculate the pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')
    
    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)
    
    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]    
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]

    # Collect the k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    
    return X_pc


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, random_state=123)
#X, y = get_dummy_data()

plt.scatter(X[y==0, 0],
            X[y==0, 1],
            color='red',
            marker='^',
            alpha=0.5
            )
plt.scatter(X[y==1, 0],
            X[y==1, 1],
            color='blue',
            marker='o',
            alpha=0.5
            )
plt.show()

# Project on the standard PCA
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0],
              X_spca[y==0, 1],
              color='red',
              marker='^',
              alpha=0.5
              )
ax[0].scatter(X_spca[y==1, 0],
              X_spca[y==1, 1],
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[1].scatter(X_spca[y==0, 0],
              np.zeros((50,1)) + 0.02,
              color='red',
              marker='^',
              alpha=0.5
              )
ax[1].scatter(X_spca[y==1, 0],
              np.zeros((50,1)) - 0.02,
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# Using radia basis function (rbf) as kernel function
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0],
              X_kpca[y==0, 1],
              color='red',
              marker='^',
              alpha=0.5
              )
ax[0].scatter(X_kpca[y==1, 0],
              X_kpca[y==1, 1],
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[1].scatter(X_kpca[y==0, 0],
              np.zeros((50,1)) + 0.02,
              color='red',
              marker='^',
              alpha=0.5
              )
ax[1].scatter(X_kpca[y==1, 0],
              np.zeros((50,1)) - 0.02,
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000,
                    random_state=123,
                    noise=0.1,
                    factor=0.2
                    )
plt.scatter(X[y==0, 0],
            X[y==0, 1],
            color='red',
            marker='^',
            alpha=0.5
            )
plt.scatter(X[y==1, 0],
            X[y==1, 1],
            color='blue',
            marker='o',
            alpha=0.5
            )
plt.show()

skicit_pca = PCA(n_components=2)
X_spca = skicit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0],
              X_spca[y==0, 1],
              color='red',
              marker='^',
              alpha=0.5              
              )
ax[0].scatter(X_spca[y==1, 0],
              X_spca[y==1, 1],
              color='blue',
              marker='o',
              alpha=0.5              
              )
ax[1].scatter(X_spca[y==0, 0],
              np.zeros((500,1))+0.02,
              color='red',
              marker='^',
              alpha=0.5              
              )
ax[1].scatter(X_spca[y==1, 0],
              np.zeros((500,1))-0.02,
              color='blue',
              marker='o',
              alpha=0.5              
              )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0],
              X_kpca[y==0, 1],
              color='red',
              marker='^',
              alpha=0.5
              )
ax[0].scatter(X_kpca[y==1, 0],
              X_kpca[y==1, 1],
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[1].scatter(X_kpca[y==0, 0],
              np.zeros((500,1)) + 0.02,
              color='red',
              marker='^',
              alpha=0.5
              )
ax[1].scatter(X_kpca[y==1, 0],
              np.zeros((500,1)) - 0.02,
              color='blue',
              marker='o',
              alpha=0.5
              )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

def rbf_kernel_pca_2(X, gamma, n_components):
    """
    RBF kernel PCA implementation
    
    Parameters
    ----------
    X: {Numpy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Tuning parameter of the RBF Kernel
    
    n_components: int
        Number of principal components to return

    Returns
    ----------
    X_pc {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset        
    """
    
    # Calculate the pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')
    
    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)
    
    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]    
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]

    # Collect the k eigenvectors (projected samples)
    # X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    
    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    
    # Collect the corresponding eigenvalues
    lamdas = [eigvals[i] for i in range(n_components)]
    
    return alphas, lamdas

X, y = make_moons(n_samples=100, random_state=123)
# X, y = get_dummy_data()
alphas, lamdas = rbf_kernel_pca_2(X, gamma=15, n_components=1)
x_new = X[25]
x_proj = alphas[25] # original projection

def project_x(x_new, X, gamma, alphas, lamdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lamdas)

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lamdas=lamdas)

plt.scatter(alphas[y==0, 0],
            np.zeros((50)),
            color='red',
            marker='^',
            alpha=0.5
            )
plt.scatter(alphas[y==1, 0],
            np.zeros((50)),
            color='blue',
            marker='o',
            alpha=0.5
            )
plt.scatter(x_proj,
            0,
            color='black',
            marker='^',
            label='original projection of point X[25]',
            s=100
            )
plt.scatter(x_reproj,
            0,
            color='green',
            marker='x',
            label='remapped point X[25]',
            s=100
            )
plt.legend(scatterpoints=1)    
plt.show()

# Using scikit-learn
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)

skicit_kpca = KernelPCA(n_components=2,
                        kernel='rbf',
                        gamma=15
                        )
X_skernpca = skicit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0],
            X_skernpca[y==0, 1],
            color='red',
            marker='^',
            alpha=0.5
            )

plt.scatter(X_skernpca[y==1, 0],
            X_skernpca[y==1, 1],
            color='blue',
            marker='o',
            alpha=0.5
            )
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
