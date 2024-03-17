import numpy as np
import matplotlib.pyplot as plt

"""
Manual GMM implementation on randomly generated points in clusters
GMM being a more "soft" clustering method which is assigning distribution to each of the point
of it belonging to a certain cluster.
    -- EM (Expectation Maximization) Algorithm is implemented
    -- after first initializing the categorical distribution, prior normal for each cluster with 0 initial mean and identity matrix as covariance
        we iterate through E and M steps and eventually converge to the true clusters
    -- several k values are checked for clusters

    
"""



# Number of samples
n_samples = 300

# Randomly generate centroids for three clusters
centroids = np.random.rand(3, 2) * 50  # Adjust the range if needed

# Generate random data around these centroids
X = np.zeros((n_samples, 2))
y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    # Randomly choose a centroid
    centroid_idx = np.random.randint(3)
    centroid = centroids[centroid_idx]
    
    # Generate data point around the chosen centroid
    X[i] = centroid + np.random.randn(2)  # Add random noise
    y[i] = centroid_idx

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Randomly Generated 2D Data with 3 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.grid(True)
plt.show()
print(X.shape)


#GMM - Expectation Maximization algorithm
import random
from scipy.stats import multivariate_normal
def gmm(X,k, num_iteration = 20):
    
    #parameter initialization
    pi_init = np.ones(k)/k
    random_indices = np.random.choice(n_samples, size=k, replace=False)
    means_init = X[random_indices]
    cov_init = [np.eye(X.shape[1]) for _ in range(k)] 
    print()
    
    gamma = np.zeros((X.shape[0], k))
    for _ in range(num_iteration):
        #Expctation - Gamma step
        for i in range(X.shape[0]):
            
            for j in range(k):
                
                mvn = multivariate_normal(mean = means_init[j], cov=cov_init[j])
                
                list_of_all = [pi_init[j]*multivariate_normal(means_init[j], cov_init[j]).pdf(X[i]) for j in range(k)]
                sum_all = sum(list_of_all)
             
                gamma[i][j] = pi_init[j]*mvn.pdf(X[i])/sum_all
                
        #print("done")
        #Maximization - parameter update step
        for j in range(k):
            Nj = sum([gamma[i][j] for i in range(X.shape[0])])
            means_init[j] = sum([gamma[i][j]*X[i] for i in range(X.shape[0])])/Nj
            cov_init[j] = sum([gamma[i][j]*np.outer((X[i] - means_init[j]),(X[i] - means_init[j])) for i in range(X.shape[0])])/Nj
            pi_init[j] = Nj/X.shape[0]
          
    return (pi_init, means_init, cov_init)           
values = gmm(X,3)
print("pi_init: ", values[0])
print("means_init: ", values[1])
print("cov_init: ", values[2])
from matplotlib.patches import Ellipse
def plot_gmm(X, means, covariances, colors):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=30, marker='.', alpha=0.5, label='Data Points')

    for i, (mean, cov, color) in enumerate(zip(means, covariances, colors)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence interval
        ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), edgecolor=color, facecolor='none')
        plt.gca().add_patch(ellipse)
        plt.scatter(mean[0], mean[1], c=color, s=100, marker='o', edgecolors='k', label=f'Cluster {i+1} Mean')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gaussian Mixture Model')
    plt.legend()
    plt.grid(True)
    plt.show()
    
plot_gmm(X,values[1], values[2], colors = ['blue', 'red', 'green'])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
"""
def plot_gmm_with_probabilities(X, means, covariances, categories, colors, markers):
    plt.figure(figsize=(8, 6))
    for category, color, marker in zip(categories, colors, markers):
        indices = np.where(X[:, 1] == category)
        plt.scatter(X[indices, 0], X[indices, 1], c=color, s=30, marker=marker, alpha=0.5, label=f'Category {category}')

    for i, (mean, cov, color) in enumerate(zip(means, covariances, colors)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence interval
        ellipse = Ellipse(mean[:2], width, height, angle=np.degrees(angle), edgecolor=color, facecolor='none')
        plt.gca().add_patch(ellipse)
        plt.scatter(mean[0], mean[1], c=color, s=100, marker='o', edgecolors='k', label=f'Cluster {i+1} Mean')

        # Compute probabilities for each data point
        mvn = multivariate_normal(mean[:2], cov)
        probabilities = mvn.pdf(X[:, :2])
        plt.scatter(X[:, 0], X[:, 1], c=probabilities, cmap='coolwarm', marker='.', alpha=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gaussian Mixture Model with Probabilities')
    plt.legend()
    plt.grid(True)
    plt.colorbar(label='Cluster Probability')
    plt.show()

# Example usage
# (Same data and parameters as before)
plot_gmm_with_probabilities(X, values[1], values[2],  categories = [1,2,3], colors = ['blue', 'red', 'green'], markers = ['o', 'o', 'o'])
"""