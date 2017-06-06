import numpy as np


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