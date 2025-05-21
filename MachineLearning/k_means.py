import numpy as np
from sklearn.cluster import KMeans

# https://www.deep-ml.com/problems/17
"""
How to pick K? Random try and compare variation with prev k value. Huge reduction in variation = k, after that there is less reduction (as u increase k clusters)
So K is called the elbow plot. KMC tries to put data into the k clusters.

Unsupervised. 
 k-Means clustering is a method used to partition n points into k clusters. 

The goal is to group similar points together and represent each group by its center (called the centroid).

    Initialization
    Use the provided initial_centroids / pick random points as your starting point. This step is already done for you in the input.

    Assignment Step
    For each point in your dataset:
        Calculate its distance to each centroid.
        Assign the point to the cluster of the nearest centroid.
        Hint: Consider creating a helper function to calculate the Euclidean distance between two points.

    Update Step
    For each cluster:
        Calculate the mean of all points assigned to the cluster.
        Update the centroid to this new mean position.
        Hint: Be careful with potential empty clusters. Decide how you'll handle them (e.g., keep the previous centroid).

    Iteration
    Repeat steps 2 and 3 until either:
        The centroids no longer change significantly (this case does not need to be included in your solution), or
        You reach the max_iterations limit.
        Hint: You might want to keep track of the previous centroids to check for significant changes.

    Result
    Return the list of final centroids, ensuring each coordinate is rounded to the nearest fourth decimal.


The distance from each point to each centroid is calculated.
Points are assigned to their nearest centroid.
Centroids are shifted to be the average value of the points belonging to it. If the centroids did not move, the algorithm is finished, else repeat.

"""


def euclid_distance(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    return np.sqrt(np.sum((x_arr - y_arr) ** 2))


def k_means_clustering(
    points: list[tuple[float, float]],
    k: int,
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:
    points = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        # for each point -> calculate distance of that point to the centroid
        for pt in points:
            dists = [euclid_distance(pt, c) for c in centroids]
            best_idx = np.argmin(dists)
            clusters[best_idx].append(pt)

        # update
        new_centroids = []
        for i in range(k):
            members = clusters[i]
            if members:
                member_arr = np.array(members)
                mean = np.mean(member_arr, axis=0)
                new_centroids.append(np.round(mean, 4))
            else:  # if mean = 0
                new_centroids.append(centroids[i])

        # check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # return centroids
    final = [tuple(float(x) for x in c) for c in centroids]
    return final


"""
using scikit learn
"""


def k_means_clustering_sklearn(
    points: list[tuple[float, float]],
    k: int,
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(points)
    return kmeans.cluster_centers_


def main():
    print(
        k_means_clustering(
            points=[(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
            k=2,
            initial_centroids=[(1, 1), (10, 1)],
            max_iterations=10,
        )
    )
    print(
        k_means_clustering_sklearn(
            points=[(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
            k=2,
            initial_centroids=[(1, 1), (10, 1)],
            max_iterations=10,
        )
    )
    # print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))


main()
