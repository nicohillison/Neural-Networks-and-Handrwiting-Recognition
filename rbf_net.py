'''rbf_net.py
Radial Basis Function Neural Network
YOUR NAME HERE
CS 251: Data Analysis Visualization, Spring 2021
'''
import numpy as np
import kmeans
import scipy.linalg


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        self.k = num_hidden_units
        self.num_classes = num_classes
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        avgDists = np.zeros(self.k)
        arr = np.zeros((data.shape[0],centroids.shape[0]))
        for c in range(self.k):
            samplesInCurrCluster = data[cluster_assignments == c]
            distAllSamplesInCluster = kmeans_obj.dist_pt_to_centroids(centroids[c], samplesInCurrCluster)
            avgDists[c] = np.mean(distAllSamplesInCluster, axis=0)
        # for i in range(len(data)):
        #     arr[i] = kmeans_obj.dist_pt_to_centroids(data[i], centroids)
        # avg = np.mean(arr, axis = 0)
        
        return avgDists

 

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmeans_obj = kmeans.KMeans(data = data)
        # Run k-means (batch version, i.e. multiple times). Set number iterations to 5
        kmeans_obj.cluster_batch(self.k, 5, verbose=False)
        #Use get methods to get: prototypes are the centroids. 
        self.prototypes = kmeans_obj.get_centroids()
        labels = kmeans_obj.get_data_centroid_labels()
        # sigmas: call avg_cluster_dist above.
        self.sigmas = self.avg_cluster_dist(data, self.prototypes, labels, kmeans_obj)

        # idx = np.random.random_integers(0 , len(data) - 1, self.k)
        # #print(idx, len(self.data))
        # self.prototypes = data[idx]

        

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        A = np.hstack([A, np.ones([A.shape[0], 1])])
        #print('this is a: ', self.A)

        c,_,_,_ = scipy.linalg.lstsq(A, y)

        return c

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        hid_mtrx = np.zeros((data.shape[0], self.k))
        for i in range(self.k):
            calc = 1 / (2*(self.sigmas[i]**2)+(1e-8))
            H = np.exp(-calc*(np.sum((data - self.prototypes[i])**2, axis = 1)))
            hid_mtrx[:,i] = H
        return hid_mtrx

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        mtrx_1 = np.ones([hidden_acts.shape[0], 1])
        h_mtrx = np.hstack((hidden_acts,mtrx_1))
        mult = h_mtrx @ self.wts
        return mult
        

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)
        l_matrix = self.hidden_act(data)
        l_matrix = np.nan_to_num(l_matrix)
        self.wts = np.zeros((self.k+1,self.prototypes.shape[1]))
        # w_mtrx = scipy.linalg.lstsq(h_mtrx, train_d[:,2])[0]
        for i in range(self.prototypes.shape[1]):
        #     print(M.shape)
        #     print(scipy.linalg.lstsq(M,train_d[:,2]))
            scip = self.linear_regression(l_matrix, y == i)
            self.wts[:,i] = scip
        

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        l_matrix = self.hidden_act(data)
        out_p = self.output_act(l_matrix)
        lst = []
        for i in range(out_p.shape[0]):
            lst.append(np.argmax(out_p[i,:]))
        return lst

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        difference = y_pred - y
        non_zero = np.count_nonzero(difference)
        accuracy = (y.shape[0]-non_zero)/y.shape[0]
        return accuracy
