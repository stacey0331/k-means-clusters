import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt


def load_iris(filepath='iris.data'):
    feat_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'class',
    ]

    df = pd.read_csv(filepath, header=None, names=feat_names)

    X = np.array(df.iloc[:, :4])

    class2idx = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    Y = np.array([class2idx[cls] for cls in df['class']])
    return X, Y, df


def load_test_example():
    X = np.array([
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
        [11, 10]
    ], dtype=float)

    Y = np.array([0, 1, 1, 1, 2])

    return X, Y


def centroids_init(X, k):
    """
    Select K points as initial centroids
    :param X: numpy.array, n_samples * n_features, data matrix
    :param k: int, number of lcusters
    :return:
    """
    # ######## ######## My Code Starts ######## ########

    centroids = [] # index of sample in X 
    np.random.seed(0)     
    for _ in range(k):
        centroids.append(X[np.random.randint(len(X[0]))])

    # ######## ######## My Code Ends ######## ########

    return centroids    # array of points, e.g. [[1,3,4],[2,4,2],[2,4,5]]


def kmeans(X, Y, k):
    """
    Select K points as initial centroids
    :param X: numpy.array, float, n_samples * n_features, data matrix
    :param Y: numpy.array, int, n_samples, ground truth
    :param k: int, number of clusters 
    :return:
    """
    centroids = centroids_init(X, k)    # array of points
    assignment = []    # array of cluster, each cluster is an array of points
    sse_lst = []  # sse in each iteration

    # ######## ######## My Code Starts ######## ########

    while True:
        sse = 0
        # Form K clusters by assigning each point to its closest centroid
        # Re-assign to clusters
        assignment = []    # empty assignment
        assginment_pts = []
        for _ in range(k):
            assginment_pts.append([])
    
        for row in X:
            min_dist = la.norm(centroids[0]-row)
            min_idx = 0
            for j in range(1,len(centroids)): # loop thru centroids
                if la.norm(centroids[j]-row) < min_dist:
                    min_dist = la.norm(centroids[j]-row)
                    min_idx = j
            sse += min_dist**2
            assignment.append(min_idx)
            assginment_pts[min_idx].append(row)

        # Re-compute the centroids (i.e., mean point) of each cluster
        for i in range(len(assginment_pts)):
            if len(assginment_pts[i]) == 0:
                continue
            centroids[i] = np.mean(assginment_pts[i], axis=0)

        # Stop if already converge
        if len(sse_lst) > 0 and sse_lst[-1] == sse:
            break  # replace this with your code
        sse_lst.append(sse)
    # ######## ######## My Code Ends ######## ########

    return centroids, assignment, sse_lst


def nmi_score(C, T, k):
    """
    Normalized Mutual Information
    :param C: numpy.array, int, cluster result
    :param T: numpy.array, int, ground truth
    :return: nmi
    """

    # ######## ######## My Code Starts ######## ########
    num_pts = len(C)
    table = np.zeros((k, len(np.unique(T))))
    for i in range(len(C)):
        table[C[i], T[i]] += 1
    PCi = np.zeros(len(table))
    for i in range(len(table)):
        PCi[i] = np.sum(table[i])/num_pts
    PTj = np.zeros(len(table[0]))
    for j in range(len(table[0])):
        PTj[j] = np.sum(table[:,j])/num_pts
    
    mutual_info = 0
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i,j] != 0:
                mutual_info += (table[i,j]/num_pts)*np.log2((table[i,j]/num_pts)/(PCi[i]*PTj[j]))
    
    # Entropy of clustering
    HC = 0
    for Ci in PCi:
        HC += (-1)*Ci*np.log2(Ci)
    
    # Entropy of partitioning
    HT = 0
    for Tj in PTj:
        HT += (-1)*Tj*np.log2(Tj)

    nmi = mutual_info/np.sqrt(HC*HT)

    # ######## ######## My Code Ends ######## ########

    return nmi


def visualize(X, assignment):
    plt.scatter(X[:, 2], X[:, 3], c=assignment)
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.title('Iris Clustering')
    plt.show()


def main():
    # load data
    X, Y, _ = load_iris()

    # ######## ######## My Code Starts ######## ########

    # visualize the clustering result with last two features
    _, assignment, sse_lst = kmeans(X, Y, 3)
    visualize(X, assignment)

    # Plot a curve of SSE w.r.t. number of iterations.
    plt.plot(sse_lst)
    plt.title('SEE w.r.t. number of iterations')
    plt.xlabel('number of iterations')
    plt.ylabel('SSE')
    plt.show()
    # Report the final SSE after K-Means converges.
    print(sse_lst)
    # Report the NMI after K-means converges.
    print(nmi_score(assignment, Y, 3))

    # Plot a curve of SSE w.r.t. k.
    sse_arr = []
    nmi_arr = []
    for k in range(2,11):
        _, assignment, sse_lst = kmeans(X, Y, k)
        sse_arr.append(sse_lst[-1])
        nmi_arr.append(nmi_score(assignment, Y, k))
    plt.plot(np.arange(2,11), sse_arr)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('SSE w.r.t. k')
    plt.show()

    # Plot a curve of NMI w.r.t. k.
    plt.plot(np.arange(2,11), nmi_arr)
    plt.xlabel('k')
    plt.ylabel('NMI')
    plt.title('NMI w.r.t. k')
    plt.show()
    # ######## ######## My Code Ends ######## ########


def test():
    X, Y = load_test_example()
    k = 2
    np.random.seed(0)  # you may want to try multiple seeds
    centroids, assignment, sse_lst = kmeans(X, Y, k)
    print(centroids, assignment, sse_lst)

    # centroids should be [10.3333, 10.3333] and [0.5, 0.5]
    # assignment should be [0, 0, 1, 1, 1] or [1, 1, 0, 0, 0]
    # last item of sse_lst should be 2.3333

    print(nmi_score(assignment, Y, k))

    # nmi should be 0.3640



if __name__ == '__main__':
    test()  # test case
    main()
