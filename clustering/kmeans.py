import numpy as np
import matplotlib.pyplot as plt
import random
import math

"""_summary_
Kmeans manual implementation on randomly generated clusters
    Kmeans being a more "hard" dicrete clustering method, which assigns a point to a cluster.
        -- we initialize centroids as many as clusters (to the points from data or randomly) ->better to do it from data
        -- then we comput distances from each point to the centroids and assign clusters to each
        -- then based on reassigned we update the centroids to the means of each clusters then go back to step2
        -- until it converges
    implemented both Lloyd's algorithm and basic approach
"""


def rand_cluster(n, c, r):
    """
    Returns n random points in a disk of radius r centered at c.
    """
    x, y = c
    points = []
    for _ in range(n):
        theta = 2 * math.pi * random.random()
        s = r * random.random()
        points.append([x + s * math.cos(theta), y + s * math.sin(theta)])
    return points

cluster_1 = rand_cluster(105, (10,10), 5)
cluster_1 = np.array(cluster_1)
cluster_2 = rand_cluster(98, (4,8), 2)
cluster_2 = np.array(cluster_2)
cluster_3 = rand_cluster(89, (-1,4), 3)
cluster_3 = np.array(cluster_3)
print(np.array(cluster_1)[:,0])
data = np.concatenate((cluster_1, cluster_2, cluster_3), axis = 0)
print(type(data))
#plt.figure(6,6)
plt.scatter(cluster_1[:,0], cluster_1[:,1], marker='o', label='Cluster 1')
plt.scatter(cluster_2[:,0], cluster_2[:,1],marker='x', label='Cluster 2')
plt.scatter(cluster_3[:,0], cluster_3[:,1], marker='D', label='Cluster 3')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data')
plt.legend()
plt.show()

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))**2

def compute_variance(clusters):
    cluster_variances = []
    for cluster in clusters:
        #print("Cluster: ", cluster)
        if len(cluster) > 1:
            cluster_var = np.var(cluster, axis=0)
            #print("Var: ", cluster_var)
            cluster_variances.append(cluster_var)
        else:
            cluster_variances.append(0)  # If only one point, variance is 0
        total_within_var = sum(cluster_variances)
        #print(total_within_var)
    return total_within_var
    
def kmeans_simple(data, n_cluster, iterations = 1):
    cluster_vars = []
    for i in range(iterations):
        centroids = random.sample(data.tolist(), n_cluster)
        labeled_data = [[] for _ in range(n_cluster)]
        for point in data:
            distances = [distance(point, centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            print("the cluster: ", closest_centroid_index)
            labeled_data[closest_centroid_index].append(point)
        #print(compute_variance(labeled_data))
        cluster_vars.append(compute_variance(labeled_data))
        for j in range(n_cluster):
            x_i = [elem[0] for elem in labeled_data[j]]
            y_i = [elem[1] for elem in labeled_data[j]]
            plt.scatter(x_i, y_i, marker='o', label=f'Cluster {j}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Original Data')
        plt.legend()
        plt.savefig(f"clusters/clustering_{i+1}.png")
        plt.close()
    best_one = np.argmin(np.array([elem[0]+elem[1] for elem in cluster_vars]))
    print(best_one)
    plt.imshow(plt.imread(f"clusters/clustering_{best_one+1}.png"))  # Display the best plot
    plt.show()
    
    return cluster_vars[best_one]

#kmeans_simple(data,3,10)

def kmeans_Lloyd(data, n_clusters, iterations):
    centroids = [random.random() for _ in range(n_clusters)]
    max_x = max([elem[0] for elem in data])
    max_y = max([elem[1] for elem in data])
    centroids = [[max_x*elem, max_y*elem] for elem in centroids]
    #print(centroids)
    for _ in range(iterations):
        z = np.zeros((len(data),n_clusters))
        data_list = data.tolist()
        for i in range(len(data_list)):
            for k in range(n_clusters):
                distances = [distance(data_list[i], centroid) for centroid in centroids]
                z[i][k] = 1 if np.argmin(distances) == k else 0
        for k in range(n_clusters):
            #print(np.where(z[:,k] == 1))
            #print("list of points: ", np.array([data_list[i] for i in np.where(z[:,k] == 1)[0]]))
            centroids[k] = [np.mean(np.array([data_list[i][0] for i in np.where(z[:,k] == 1)[0]])), np.mean(np.array([data_list[i][1] for i in np.where(z[:,k] == 1)[0]]))]
    labeled_data = [[] for _ in range(n_clusters)]
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        print("the cluster: ", closest_centroid_index)
        labeled_data[closest_centroid_index].append(point)
    #print(compute_variance(labeled_data))
    #cluster_vars.append(compute_variance(labeled_data))
    for j in range(n_clusters):
        x_i = [elem[0] for elem in labeled_data[j]]
        y_i = [elem[1] for elem in labeled_data[j]]
        plt.scatter(x_i, y_i, marker='o', label=f'Cluster {j}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Data')
    plt.legend()
    plt.show()
    
    #print(centroids)

    return centroids

kmeans_Lloyd(data, 3, 20)
    
