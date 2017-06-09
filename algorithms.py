import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()


class NonparametricRegression(object):
    def __init__(self):
        """Initializing the quantiles to be .01-.99"""

        self.quantiles = np.arange(0.01,1,.01).tolist()
        self.num_quantiles = len(self.quantiles)


    def cross_val(self, x_train, y_train):
        """Finding hyperparameters using a grid search and cross validation."""
        
        p = x_train.shape[1]

        feature_num_grid = [p/4, p/2, (p*3)/4]
        cluster_grid = [p/4, p/2, (p*3)/4]
        iter_grid = [100]
        lamb_grid = [.01, .1, .5, 1]

        param_loss = []

        for params in itertools.product(feature_num_grid, cluster_grid, lamb_grid, iter_grid):
            self.p = params[0]
            self.k = params[1]
            self.lamb = params[2]
            self.iters = params[3]

            total_samples = len(x_train)

            part_sample = total_samples/5

            splits = [range(part_sample), range(part_sample, part_sample*2), 
                      range(part_sample*2, part_sample*3), range(part_sample*3, part_sample*4),
                      range(part_sample*4, total_samples)]

            folds = [(splits[0]+splits[1]+splits[2]+splits[3], splits[4]), 
                     (splits[0]+splits[1]+splits[2]+splits[4], splits[3]), 
                     (splits[0]+splits[1]+splits[4]+splits[3], splits[2]), 
                     (splits[0]+splits[4]+splits[2]+splits[3], splits[1]), 
                     (splits[4]+splits[1]+splits[2]+splits[3], splits[0])]


            total_loss = 0.0

            for fold in folds:
                x_tr = x_train[fold[0]]
                y_tr = y_train[fold[0]]
                x_t = x_train[fold[1]]
                self.y_test = y_train[fold[1]]

                self.train(x_tr, y_tr)

                x_t = x_t[:,self.j_prime[:self.p]]
                self.phi_test = self.rbf_feature_generation(x_t)

                self.predict_member()

                loss = self.loss()
                total_loss += loss

            param_loss.append((total_loss, params))

        best_params = sorted(param_loss, key=lambda x: x[0])[0][1]

        return best_params


    def fit(self, x_train, y_train):
        """Fitting the model once parameters have been chosen through CV.

        :param x_train: m x p numpy array of covariates.
        :param y_train: m x 1 numpy array of target variables.
        """

        best_params = self.cross_val(x_train, y_train)

        self.p = best_params[0]
        self.k = best_params[1]
        self.lamb = best_params[2]
        self.iters = best_params[3]

        self.train(x_train, y_train)


    def predict(self, x_test, y_test):
        """Prediction on the test set after training a model.

        :param x_test: m x p numpy array of covariates.
        :param y_test: m x 1 numpy array of target variables.
        """

        # Using the input features chosen from cross validation.
        x_test = x_test[:, self.j_prime[:self.p]]

        self.y_test = y_test

        # Generating RBF features for the testing set.
        self.phi_test = self.rbf_feature_generation(x_test)

        self.predictions = self.phi_test.dot(self.theta)

        self.m = self.predictions.shape[0]

        # Generating the score for the test set.
        self.loss = self.loss()


    def train(self, x_train, y_train):
        """Training the 3 stage model for quantile regression."""

        self.x_train = x_train
        self.y_train = y_train

        self.m = x_train.shape[0]
        self.p = x_train.shape[1]

        self.forward_stepwise_selection_lm()

        self.x_train = self.x_train[:, self.j_prime[:self.p]]

        self.k_means()

        self.phi_train = self.rbf_feature_generation(self.x_train)

        self.admm_quantile_reg()


    def loss(self):
        """Calculating the loss using the scoring function from GEFCom2014"""

        total_loss = 0.0

        for quantile in range(self.num_quantiles):

            curr_quantile = self.predictions[:,quantile]

            curr_loss = 0.0

            for i in range(len(curr_quantile)):

                if self.y_test[i] < curr_quantile[i]:
                    curr_loss += (1 - self.quantiles[quantile]/100.)*(curr_quantile[i] - self.y_test[i])

                else:
                    curr_loss += self.quantiles[quantile]/100. * (self.y_test[i] - curr_quantile[i])

            curr_loss /= float(len(curr_quantile))

            total_loss += curr_loss

        total_loss /= float(self.num_quantiles)

        return total_loss


    def predict_member(self):
        """Prediction only called internally in the cross validation."""

        self.predictions = self.phi_test.dot(self.theta)


    def forward_stepwise_selection_lm(self):
        """Choose features to use by forward stepwise selection"""
        
        # Begin with an intercept term.    
        self.x_prime = np.ones((self.m, 1))

        # To hold the order of the index added.
        self.j_prime = []

        # To hold the error at each step of adding a covariate.
        self.e_prime = []

        p = self.x_train.shape[1]
        
        for i in range(p):
            
            # Computing the LOOCV by adding each feature.
            errors = [self.efficient_LOOCV(np.hstack((self.x_prime, self.x_train[:, j, None]))) 
                      if j not in self.j_prime else float('inf') for j in xrange(p)] 
            
            self.e_prime.append(min(errors))
            
            # Finding the feature index that minimized the LOOCV by adding the feature.
            self.j_prime.append(np.argmin(np.array(errors)))

            # Adding the best feature to the new feature vector to keep.
            self.x_prime = np.hstack((self.x_prime, self.x_train[:, self.j_prime[-1], None]))
        
        # Removing the intercept term.
        self.x_prime = self.x_prime[:, 1:]
        

    def efficient_LOOCV(self, x):
        """Compute the LOOCV for least squares for forward stepwise selection.
        
        :param x: Numpy array like of the feature space.
        
        :return loocv: The leave one out cross validation error.
        """

        # Smoothing matrix for the linear smoother.
        hat_matrix = x.dot(np.linalg.pinv(x.T.dot(x))).dot(x.T)
        hat_diag = np.diagonal(hat_matrix).reshape((-1,1))
        
        f_hat = self.least_squares(x)
        
        residuals = self.y_train - f_hat
        
        smoothing = 1. - hat_diag

        loocv_vector = residuals/smoothing

        loocv = 1/float(self.m) * np.dot(loocv_vector.T, loocv_vector)
            
        return float(loocv)


    def least_squares(self, x):
        """Compute the least squares predictions.
        
        :param x: Numpy array like of the training feature space.
        
        :return predictions: Numpy array of the predictions for the data.
        """
        
        beta = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(self.y_train))
        
        predictions = x.dot(beta)
        
        return predictions


    def k_means(self, num_restarts=10):
        """K-means++ algorithm for RBF center and bandwidth choice. 
        
        This function calls the k-means++ algorithm with multiple restarts 
        chooses the start that minimized the loss.
        
        :param num_restarts: Integer number of times to restart the algorithm to 
        avoid local min. The iteration results that minimized the loss is chosen.
        """
        
        results = []

        # Running k-means with default of 15 restarts and saving results each run.
        for restart in range(num_restarts):
            results.append(self.k_means_alg())      
        
        loss_results = np.array([result[0] for result in results])

        # Finding the start index that minimized the loss.
        best_loss_index = np.argmin(loss_results)
        
        # Results that minimized the loss.
        best_result = results[best_loss_index]

        # k x q matrix with each row being a center for an RBF basis.
        self.mu = best_result[1]

        # k x 1 matrix with each row being the bandwidth for an RBF basis.
        self.sigma = best_result[2]
        

    def k_means_alg(self):
        """K-means++ algorithm for RBF feature creation. 
        
        This function implements the k-means++ algorithm, which is an approach to
        spread out the initial cluster centers by choosing a first cluster center at 
        random, then proceeding to choose centers sampled from the data with 
        probability proportional to the squared distance to the points existing 
        closest center.
                
        :return loss: Final result for the loss function.
        :return means: Numpy array with each row a centroid of a cluster. 
        :return bandwidths: Numpy array with each row the selected bandwidth for
        for the corresponding cluster.
        """
                
        # Seed numpy random number generator.
        np.random.seed()
        
        # Choosing the first cluster centroid uniformly at random.
        init = np.random.choice(range(self.m), 1, replace=False).tolist()
        means = self.x_train[init]
        
        # Choosing the rest of the cluster centroid initializations.
        for i in range(1, self.k):
            
            # Finding minimum squared distance for each point to an existing center.
            squared_dists = np.array([[np.linalg.norm(self.x_train[j]-means[l])**2 
                                     for l in xrange(i)] for j in xrange(self.m)])

            dist_mins = squared_dists.min(axis=1)
            
            # Sampling with probability proportional to the minimum squared distance.
            prob_weights = dist_mins/dist_mins.sum()
            sample = np.random.multinomial(1, prob_weights).tolist()
            sample_choice = sample.index(1)
            
            means = np.vstack((means, self.x_train[sample_choice]))
        
        # K-means algorithm.
        old_labels = np.nan * np.zeros(self.m)
        converged = False
        
        while not converged:

            # Assigning points to clusters.
            squared_dists = [[np.linalg.norm(self.x_train[i]-means[l])**2 
                             for l in xrange(self.k)] for i in xrange(self.m)]

            labels = np.argmin(np.array(squared_dists), axis=1)
                    
            # Check for convergence, given by no labels changing.
            if np.array_equal(labels, old_labels):
                converged = True
            else:
                pass
            
            old_labels = labels
                
            # Recalculating the cluster centroids.
            means = np.array([self.x_train[np.where(labels == i)[0]].mean(axis=0).tolist() 
                             for i in xrange(self.k)])    
        
        # Calculating the total loss.
        squared_dists = [[np.linalg.norm(self.x_train[i]-means[l])**2 for l in xrange(self.k)] for i in xrange(self.m)]
        loss = np.array(squared_dists).min(axis=1).sum()
        
        # Calculating the bandwidth for RBF features using median trick.
        bandwidths = [[np.linalg.norm(means[i] - means[j]) for j in xrange(self.k) if i != j] for i in xrange(self.k)]
        bandwidths = np.median(np.array(bandwidths), axis=1).reshape((-1,1))
        
        return loss, means, bandwidths


    def rbf_feature_generation(self, x):
        """Create radial basis function features using squared exponential basis functions.
        
        :param x: Numpy array of feature space to transform to RBF features.
        
        :return phi: Numpy array of the feature space transformed to a RBF space.
        """
        
        m = x.shape[0]

        # Squared exponential bias function.
        rbf = lambda x, mu, sigma: math.exp(-(np.linalg.norm(x-mu)**2)/(2.*sigma**2))
        
        phi = [[rbf(x[i], self.mu[j], self.sigma[j]) for j in xrange(self.k)] for i in xrange(m)]
        
        # m x k array of the RBF feature space.
        phi = np.array(phi)
        
        return phi


    def admm_quantile_reg(self):
        """ADMM for multiple quantile regression."""

        step = 1

        self.theta = np.zeros((self.k, self.num_quantiles))
        U = np.zeros((self.m, self.num_quantiles))
        Z = np.zeros((self.m, self.num_quantiles))

        # k x k array of the Cholesky factorization.
        L = np.linalg.cholesky(self.phi_train.T.dot(self.phi_train) + self.lamb/float(step)*np.identity(self.k))
        L_inv = np.linalg.inv(L)

        ones = np.ones((self.num_quantiles, 1))
        
        gamma = 1/float(step)
        
        # Soft thresholding function for ADMM optimization.
        soft_thresholding = lambda x, gamma, alpha: np.maximum(0, x - gamma*alpha) \
                                                    + np.minimum(0, x - gamma*(alpha - 1)) 
        
        # ADMM updates applied for every quantile in each iteration.
        for i in range(self.iters):
            self.theta = L_inv.T.dot(L_inv).dot(self.phi_train.T).dot(self.y_train.dot(ones.T) + Z - U)
            
            threshold_input = self.phi_train.dot(self.theta) - self.y_train.dot(ones.T) + U
            
            # Update for z has to be applied with each quantile value.
            Z = [soft_thresholding(threshold_input[:, i], gamma, self.quantiles[i]) for i in range(self.num_quantiles)]
            
            Z = np.vstack(Z).T
            
            U = U + self.phi_train.dot(self.theta) - self.y_train.dot(ones.T) - Z


    def plot_results(self):
        """Visualizing the quantiles versus the true function using an envelope method."""

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        ax = plt.axes(xlim=(0, self.m), ylim=(0, max(np.max(self.y_test), np.max(self.predictions))+.5))

        plt.plot(range(self.m), self.predictions[:,0], linestyle='-', linewidth=1, color='blue', alpha=.3)
        plt.fill_between(range(self.m), self.predictions[:,0], self.predictions[:,19], facecolor='blue', alpha=.3)

        plt.plot(range(self.m), self.predictions[:,19], linestyle='-', linewidth=1, color='blue', alpha=.6)
        plt.fill_between(range(self.m), self.predictions[:,19], self.predictions[:,39], facecolor='blue', alpha=.6)

        plt.plot(range(self.m), self.predictions[:,39], linestyle='-', linewidth=1, color='blue', alpha=.9)
        plt.fill_between(range(self.m), self.predictions[:,39], self.predictions[:,59], facecolor='blue', alpha=.9)

        plt.plot(range(self.m), self.predictions[:,59], linestyle='-', linewidth=1, color='blue', alpha=.6)
        plt.fill_between(range(self.m), self.predictions[:,59], self.predictions[:,79], facecolor='blue', alpha=.6)

        plt.plot(range(self.m), self.predictions[:,79], linestyle='-', linewidth=1, color='blue', alpha=.3)
        plt.fill_between(range(self.m), self.predictions[:,79], self.predictions[:,98], facecolor='blue', alpha=.3)

        plt.plot(range(self.m), self.predictions[:,98], linestyle='-', linewidth=1, color='blue', alpha=.3)

        plt.plot(range(self.m), self.y_test, '-o', color='gold', linewidth=1.5)

        plt.setp(ax.get_xticklabels(), fontsize=22, rotation=60)
        plt.setp(ax.get_yticklabels(), fontsize=22)

        plt.title('Nonparametric Quantile Regression', fontsize=24)

        plt.tight_layout()
            
        sns.reset_orig()

        plt.show()
            