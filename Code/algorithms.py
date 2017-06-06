import numpy as np
import multiprocessing
import functools 

def least_squares(x_train, y_train, x_test, intercept=True):
    """Compute the least squares predictions.
    
    :param x_train: Numpy array like of the training feature space.
    :param y_train: Numpy array like of the training labels.
    :param x_test: Numpy array like of the feature space to predict on.
    :param intercept: Boolean indicating whether or not an intercept has been
    added to the feature spaces yet.
    
    :return predictions: Numpy array of the predictions for the training data.
    """
    
    if intercept:
        pass
    else:
        n_tr = len(x_train)
        n_t = len(x_test)

        x_train = np.hstack((np.ones((n_tr, 1)), x_train))
        x_test = np.hstack((np.ones((x_t, 1)), x_test))
    
    beta = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T.dot(y_train))
    
    predictions = x_test.dot(beta)
    
    return predictions


def lstqr_hat_matrix(x_train, y_train, intercept=True):
    """Calculating the smoothing matrix S.
    
    :param x_train: Numpy array like of the training feature space.
    :param y_train: Numpy array like of the training labels.
    :param intercept: Boolean indicating whether or not an intercept has been
    added to the feature spaces yet.
    
    :return hat_matrix: Numpy array of the smoothing matrix S for the model.
    """
        
    if intercept:
        pass
    else:
        n_tr = len(x_train)
        x_train = np.hstack((np.ones((n_tr, 1)), x_train))
        
    hat_matrix = x_train.dot(np.linalg.pinv(x_train.T.dot(x_train))).dot(x_train.T)
    
    return hat_matrix


def efficient_LOOCV(x_train, y_train, learning_alg, smoothing_alg):
    """Compute the LOOCV for a training and testing set of data.
    
    :param x_train: Numpy array like of the feature space.
    :param y_train: Numpy array like of the training labels.
    :param learning_alg: Function to compute the predictions for the training 
    data. This learning algorithm must be a linear smoother.
    :param smoothing_alg: Function to compute the smoothing matrix.
    
    :return loocv: The leave one out cross validation error.
    """
    
    n = float(len(x_train))
    
    hat_matrix = smoothing_alg(x_train, y_train)
    hat_diag = np.diagonal(hat_matrix).reshape((-1,1))
    
    f_hat = learning_alg(x_train, y_train, x_train)
    
    residuals = y_train - f_hat
    
    smoothing = 1 - hat_diag

    loocv_vector = residuals/smoothing

    loocv = 1/n * np.dot(loocv_vector.T, loocv_vector)
        
    return float(loocv)


def forward_stepwise_selection_lm(x, y, q, learning_alg, smoothing_alg):
    """Choose features to use by forward stepwise selection for a linear model.
    
    :param x: Numpy array like of the feature space.
    :param y: Numpy array like of the labels for the data.
    :param q: Integer of the number of features to select.
    :param learning_alg: Function to compute the predictions for the training 
    data. This learning algorithm must be a linear smoother.
    :param smoothing_alg: Function to compute the smoothing matrix.
    
    :return x_prime: Numpy array of the new feature space selected.
    :return j_prime: List of the indexes of the feature space that were kept.
    """
    
    n = x.shape[0]
    p = x.shape[1]
    
    # Begin with an intercept term.    
    x_prime = np.ones((n, 1))
    j_prime = []
    
    for i in range(q):
        
        # Computing the loocv by adding each feature.
        errors = [efficient_LOOCV(np.hstack((x_prime, x[:, j, None])), y, learning_alg, smoothing_alg) 
                  if j not in j_prime else float('inf') for j in xrange(p)] 
        
        # Finding the feature index that minimized the loocv by adding the feature.
        j_prime.append(np.argmin(np.array(errors)))

        # Adding the best feature to the new feature vector to keep.
        x_prime = np.hstack((x_prime, x[:, j_prime[-1], None]))
    
    x_prime = x_prime[:, 1:]
    
    return x_prime, j_prime


def k_means(x, k, num_restarts=15):
    """K-means++ algorithm for RBF feature creation. 
    
    This function calls the k-means++ algorithm with multiple restarts 
    concurrently and chooses the start that minimized the loss.
    
    :param x: Feature space to create the RBF features from.
    :param k: Integer number of clusters to use for the algorithm.
    :param num_restarts: Integer number of times to restart the algorithm to 
    avoid local min. The iteration results that minimized the loss is chosen.
    
    :return loss: Final result for the loss function.
    :return means: Numpy array with each row a centroid of a cluster. 
    :return bandwidths: Numpy array with each row the selected bandwidth for
    for the corresponding cluster.
    """
    
    pool = multiprocessing.Pool()
    
    func = functools.partial(k_means_alg, x, k)
    results = pool.map(func, range(num_restarts))    
    
    pool.close()
    pool.join()    
    
    loss_results = np.array([result[0] for result in results])
    best_loss_index = np.argmin(loss_results)
    
    loss, means, bandwidths = results[best_loss_index]
    
    return loss, means, bandwidths


def k_means_alg(x, k, restart_num):
    """K-means++ algorithm for RBF feature creation. 
    
    This function implements the k-means++ algorithm, which is an approach to
    sproad out the initial cluster centers by choosing a first cluster center at 
    random, then proceeding to choose centers sampled from the data with 
    probability proportional to the squared distance to the points existing 
    closest center.
    
    :param x: Feature space to create the RBF features from.
    :param k: Integer number of clusters to use for the algorithm.
    :param restart_num: Count of which restart is being called.
    
    :return loss: Final result for the loss function.
    :return means: Numpy array with each row a centroid of a cluster. 
    :return bandwidths: Numpy array with each row the selected bandwidth for
    for the corresponding cluster.
    """
    
    n = len(x)
    
    # Seed numpy random number generator.
    np.random.seed()
    
    # Choosing the first cluster centroid uniformly at random.
    init = np.random.choice(range(n), 1, replace=False).tolist()
    means = x[init]
    
    # Choosing the rest of the cluster centroid initializations.
    for i in range(1,k):
        
        # Finding minimum squared distance for each point to an existing center.
        squared_dists = np.array([[np.linalg.norm(x[j]-means[l])**2 for l in xrange(i)] for j in xrange(n)])
        dist_mins = squared_dists.min(axis=1)
        
        # Sampling with probability proportional to the minimum squared distance.
        prob_weights = dist_mins/dist_mins.sum()
        sample = np.random.multinomial(1, prob_weights).tolist()
        sample_choice = sample.index(1)
        
        means = np.vstack((means, x[sample_choice]))
    
    # K-means algorithm.
    old_labels = np.nan * np.zeros(n)
    converged = False
    
    while not converged:

        # Assigning points to clusters.
        squared_dists = [[np.linalg.norm(x[i]-means[l])**2 for l in xrange(k)] for i in xrange(n)]
        labels = np.argmin(np.array(squared_dists), axis=1)
                
        # Check for convergence, given by no labels changing.
        if np.array_equal(labels, old_labels):
            converged = True
        else:
            pass
        
        old_labels = labels
            
        # Recalculating the cluster centroids.
        means = np.array([x[np.where(labels == i)[0]].mean(axis=0).tolist() for i in xrange(k)])    
    
    # Calculating the total loss.
    squared_dists = [[np.linalg.norm(x[i]-means[l])**2 for l in xrange(k)] for i in xrange(n)]
    loss = np.array(squared_dists).min(axis=1).sum()
    
    # Calculating the bandwidth for RBF features using median trick.
    bandwidths = [[np.linalg.norm(means[i] - means[j]) for j in xrange(k) if i != j] for i in xrange(k)]
    bandwidths = np.median(np.array(bandwidths), axis=1).reshape((-1,1))
    
    return loss, means, bandwidths

