#Unsupervised Learning 
#Dimensionality reduction - PCA, SVD

######################################
# THE REASON MANUAL AND LIBRARY IMPLEMENTATIONS DIFFER is that library pca doesn't divide by standard deviation but we do
"""
Singular value decomposition (SVD) and principal component analysis (PCA) are two eigenvalue methods 
used to reduce a high-dimensional data set into fewer dimensions while retaining important information.
    PCA manual implementation:
        -- data normalization
        -- covariance matrix calculation
        -- eigenvalue decomposition -> QR method chosen: Gramschmidt calculated first, 
                                        then iterativly eigenvalues and eigenvectors calculated
        -- Data truncation
        -- Visualization based on first three PCs
    SVD manual implementation:
        -- based on PCA results

"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

data = pd.read_csv(f"./dimensionality_reduction/Country-data.csv")

matrix = data.iloc[:, 1:].values
singular_vector = np.ones(len(matrix))

matrix_mean = np.array(np.dot(np.transpose(matrix),singular_vector) / len(matrix))

normalized_matrix = matrix - matrix_mean

cov_matrix0 = np.cov(normalized_matrix, rowvar=False)

#Hand calculated covariance matrix
"""
cols = data.shape[1]-1
cov_matrix = np.zeros((cols, cols))
cov_matrix_easy = np.dot(np.transpose(normalized_matrix), normalized_matrix) / (len(normalized_matrix)) 
print("Cov easy: ", cov_matrix_easy)

for i in range(cols):
    for j in range(cols):
        if i == j:
            cov_matrix[i][j] = np.var(normalized_matrix[:,i])
        else:
            cov_matrix[i][j] = np.cov(normalized_matrix[:,i], normalized_matrix[:,j])[0,1]
            
cov_matrix_eff = np.cov(normalized_matrix, rowvar=False)"""
#print("Cov: ", cov_matrix)

from tabulate import tabulate

def gram_schmidt_qr(matrix):
    m, n = matrix.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        v = matrix[:, j].astype(np.float64)  # Convert to float
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].T, matrix[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def eigen_qr_simple(A, iterations=500000):
    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        Q, R = gram_schmidt_qr(Ak)
        Ak = R @ Q
        QQ = QQ @ Q
        # we "peek" into the structure of matrix A from time to time
        # to see how it looks
        """if k % 10000 == 0:
            print("A", k, "=")
            print(tabulate(Ak))
            print("\n")"""
    return Ak, QQ

evals, evecs = np.linalg.eigh(cov_matrix0)
print("Eigenvalue:", evals)
list_eigh =[]
for i in range(len(evals)):
    list_eigh.append((evals[i], evecs[i]))
    
list_eigh = sorted(list_eigh, key=lambda x:x[0])
list_eigh = list_eigh[::-1]
evecs_sorted = [elem[1] for elem in list_eigh]
evals_sorted = [elem[0] for elem in list_eigh]
truncated_data = np.dot(normalized_matrix, np.transpose(np.array(evecs_sorted[:3])))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(truncated_data[:,0], truncated_data[:,1], truncated_data[:,2])
    
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('Scatter Plot of Truncated Data (3D)')
plt.show()

from sklearn.decomposition import PCA
data = pd.read_csv(f"./dimensionality_reduction/Country-data.csv")

# Separate features from the target variable if applicable
X = data.drop(columns=["country"])
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])
    
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('Scatter Plot of Truncated Data (3D)')
plt.show()


#Class implementation
import sys


class PCA:
    def __init__(self, dataset_path):
        self.data_path = dataset_path
        
    def preprocess(self):
        data = pd.read_csv(self.data_path)
        matrix = data.iloc[:,1:].values
        self.rows = data.shape[0]
        self.columns = data.shape[1]-1
        singular_vector = np.ones(len(matrix))
        matrix_mean = np.transpose(matrix)*singular_vector / len(matrix)
        self.normalized_matrix = matrix - matrix_mean
        
    def covariance_matrix(self):
        matrix = self.normalized_matrix
        cov_matrix = np.zeros((self.columns, self.columns))
        for i in range(self.columns):
            for j in range(self.columns):
                if i == j:
                    cov_matrix[i][j] = np.var(matrix[:,i])
                else:
                    cov_matrix[i][j] = np.cov(matrix[:,i], matrix[:,j])[0,1]
        self.cov_mat = cov_matrix
        
    def eigenvalues(self, show = False):
        from tabulate import tabulate

        def gram_schmidt_qr(matrix):
            m, n = matrix.shape
            Q = np.zeros((m, n), dtype=np.float64)
            R = np.zeros((n, n), dtype=np.float64)

            for j in range(n):
                v = matrix[:, j].astype(np.float64)  # Convert to float
                for i in range(j):
                    R[i, j] = np.dot(Q[:, i].T, matrix[:, j])
                    v -= R[i, j] * Q[:, i]

                R[j, j] = np.linalg.norm(v)
                Q[:, j] = v / R[j, j]

            return Q, R

        def eigen_qr_simple(A, iterations=500000):
            Ak = np.copy(A)
            n = A.shape[0]
            QQ = np.eye(n)
            for k in range(iterations):
                Q, R = gram_schmidt_qr(Ak)
                Ak = R @ Q
                QQ = QQ @ Q
                # we "peek" into the structure of matrix A from time to time
                # to see how it looks
                if show:
                    if k % 10000 == 0:
                        print("A", k, "=")
                        print(tabulate(Ak))
                        print("\n")
            return Ak, QQ
        eigenvalue_approximation_matrix, eigenvector_approximation_matrix = eigen_qr_simple(self.cov_mat)
        eigenvalues = np.diag(eigenvalue_approximation_matrix)
        eigenvalue_matrix = np.zeros((len(eigenvalues), len(eigenvalues)))
        np.fill_diagonal(eigenvalue_matrix, eigenvalues)
        
        
        
# SVD
#Since SVD and PCA are equivalent I first came up with a way to decompose and truncate with PCA,
#Now I will get SDV matrices using PCA

eigenvalue_approximation_matrix, eigenvector_approximation_matrix = eigen_qr_simple(cov_matrix0)
eigenvalues = np.diag(eigenvalue_approximation_matrix)
eigenvalue_matrix = np.zeros((len(eigenvalues), len(eigenvalues)))
np.fill_diagonal(eigenvalue_matrix, eigenvalues)
Sigma = np.sqrt(eigenvalue_matrix)
V = eigenvector_approximation_matrix
U = np.dot(np.dot(normalized_matrix, V), np.linalg.inv(Sigma))