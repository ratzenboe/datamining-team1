"""

Modified version with comments; only for testing

"""



import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import copy
import time


class PreDeCon:
    def __init__(self, data, epsilon, delta, mu, lamb, kappa):
        """
        :param data: data array: numpy ndarray
        :param epsilon: Defines region of interest for each point in data set; radius parameter for variance calcs
        :param delta: Threshold parameter for variance along attriubte determining the subspace preference dim.
        :param mu: Minimum number of points (similar to DBSCAN)
        :param lamb: maximum subspace preference dimensionaliy to be considered a core point
        :param kappa: subspace scaling factor; usually k>>1
        """
        # data and hyperparameters.
        self.data = data
        self.epsilon = epsilon
        self.delta = delta
        self.mu = mu
        self.lamb = lamb
        self.kappa = kappa

        # Number of points and dimensions
        self.nb_points, self.nb_dimensions = data.shape
        print('0')

        # Fill member variables at initialization
        self.epsilon_neighbourhoods = self.create_epsilon_neighbourhoods()
        print('1')
        self.attribute_variances = self.create_variances_along_attributes()
        print('2')
        self.subspace_preference_dimensionalities = self.create_subspace_preference_dimensionality()
        print('3')
        self.subspace_preference_vectors = self.create_subspace_preference_vectors()
        print('4')
        # problem after 4
        start = time.time()
        self.pref_weighted_similarity_measures = self.create_preference_weighted_similarity_matrix()
        end = time.time()
        print('5, time = ', end - start)
        self.preference_weighted_neighbourhoods = self.create_preference_weighted_epsilon_neighbourhood()
        print('6')
        self.preference_weighted_core_points = self.create_preference_weighted_core_points()
        print('7')
        
        # Cluster label information
        self.labels_ = np.full(shape=(self.nb_points,), fill_value=np.nan)
        print('8')

    def create_epsilon_neighbourhoods(self):
        """Calculate epsilon-neighbourhood for each data point
        :return: numpy ndarray where each row contains the indices of points that
                 are in the neighborhood of a certain point
        """
        neigh = NearestNeighbors(radius=self.epsilon)
        neigh.fit(self.data)
        _, neigh_idx = neigh.radius_neighbors(self.data)
        return neigh_idx

    def create_variances_along_attributes(self):
        """Calculate variance across attributes
        :return: numpy ndarray: rows=data points, columns= variance in corresponding attribute
        """
        attribute_vars = np.empty(shape=(0, self.nb_dimensions))
        for idx in range(self.nb_points):
            # Calculate attribute variances
            var_attr = np.sum((self.data[idx] - self.data[self.epsilon_neighbourhoods[idx]]) ** 2, axis=0) / len(
                self.epsilon_neighbourhoods[idx])
            attribute_vars = np.vstack([attribute_vars, var_attr])  # Save them in numpy ndarray
        return attribute_vars

    def create_subspace_preference_dimensionality(self):
        """Calcuate subspace preference dimensionaliy from the variance across attributes
        :return: numpy 1D array: one dimensionality value for each data point
        """
        # For each point compute number of dimensions that have a lower variance then delta
        spd = np.count_nonzero(self.attribute_variances < self.delta, axis=1)
        return spd

    def create_subspace_preference_vectors(self):
        """For each point and attribute we get a weight (thus a vector for multiple attributes),
         where the weight is 1 if the variance along the axis is large than delta, else the weight is kappa
        :return: numpy ndarray: rows=data points, columns= weight in corresponding attribute
        """
        # Define weight vector: has the same dimensionaliy as the attributes variances & consists of 1's and kappas
        weight_vec = np.ones_like(self.attribute_variances, dtype=float)
        # set weights whose attributes are smaller than delta to kappa
        weight_vec[self.attribute_variances <= self.delta] = self.kappa
        return weight_vec

    def create_preference_weighted_similarity_matrix(self):
        """Calcuate similarity matrix of all points in the data set (fast implementation but O(2*n^2*d) memory complexit)
        :return: numpy ndarray: symmetrical distance matrix using the weighted similarity measure
        """
        # Create distance matrix for each attribute
        dist_mtrx_cols = np.empty(shape=(self.nb_points, self.nb_points, self.nb_dimensions))
        print('5.1')
        # print('self.nb_dimensions = ', self.nb_dimensions)
        
        start = time.time()
        for i in range(self.nb_dimensions):
            dist_mtrx_cols[:, :, i] = cdist(self.data[:, i, None], self.data[:, i, None], 'euclidean') ** 2  
        end = time.time()
        print('loop after 5.1 run time = ', end - start)
        # it took during one run 72.66661500930786 sec with Sebastian's parameters/10
        # it took during one run 119.07423210144043 sec with Sebastian's parameters
        # it took during one run 69.43253707885742 sec with parameters: epsilon=0.05; delta=0.025; mu=5; lamb=0.01; kappa=500
        # it took during one run 54.549379110336304 sec with Sebastian's parameters and: epsilon=0.1
        # it took during one run 47.692484855651855 sec with Sebastian's parameters and: delta=0.025
        # it took during one run 71.068363904953 sec with Sebastian's parameters and: epsilon=0.1, delta=0.025
        
        # it took during one run 45.503710985183716 sec with Sebastian's parameters and shortest_path instead of gram: 
        # it took during one run ? sec with Sebastian's parameters and: 
        print('5.2')
        
        # Can multiply each column and row with weight vector; the final matrix is the maximum of those oparations
        start = time.time()
        dist_mtrx_rows = copy.deepcopy(dist_mtrx_cols)
        end = time.time()
        print('loop after 5.2 run time = ', end - start)
        # it took during one run 72.06935811042786 sec with Sebastian's parameters/10
        # it took during one run 102.9776132106781 sec with Sebastian's parameters
        # it took during one run 79.9322760105133 sec with parameters: epsilon=0.05; delta=0.025; mu=5; lamb=0.01; kappa=500
        #
        # it took during one run 69.26668190956116 sec with Sebastian's parameters and: delta=0.025
        
        # it took during one run 68.9875500202179 sec with Sebastian's parameters and shortest_path instead of gram: 
        print('5.3')
        
        # problem in loop after 5.3, it went through 1 iteration for i=0 and then kernel died error was shown
        
        # Multiply each row by the corresponding weight
        start = time.time()
        for i in range(self.nb_dimensions): 
            dist_mtrx_rows[:, :, i] *= self.subspace_preference_vectors[:, i, None]
        end = time.time()
        print('loop after 5.3 run time = ', end - start)
        print('5.4')
        
        # Multiply each column by the corresponding weight
        start = time.time()
        for i in range(self.nb_dimensions): 
            dist_mtrx_cols[:, :, i] *= self.subspace_preference_vectors[:, i]
        end = time.time()
        print('loop after 5.4 run time = ', end - start)
        print('5.5')
        
        # Compute full distance matrix by summing over the 3rd axis and then taking the square root
        dist_mtrx_rows = np.sqrt(np.sum(dist_mtrx_rows, axis=2))
        print('5.6')
        dist_mtrx_final = np.sqrt(np.sum(dist_mtrx_cols, axis=2))
        print('5.7')
        # To symmetrize the distance matrix choose the largest distance
        dist_mtrx_final[dist_mtrx_rows > dist_mtrx_final] = dist_mtrx_rows[dist_mtrx_rows > dist_mtrx_final]
        print('5.8')
        return dist_mtrx_final

    def create_preference_weighted_epsilon_neighbourhood(self):
        """New neighborhood definition under the weighted similarity measure
        :return: List of numpy 1D arrays containing indices of points in the preference weighted eps. nbh
        """
        arr_indices = np.arange(self.nb_points)
        weighted_eps_neighbh = [arr_indices[dist_row < self.epsilon]
                                for dist_row in self.pref_weighted_similarity_measures]
        return weighted_eps_neighbh

    def create_preference_weighted_core_points(self):
        """ For each point in the data set test if the conditions 1&2 to a preference weighted core point are satisfied:
            1. Subspace dimensionality is smaller or equal than lambda
            2. The number of preference weighted epsilon neighbors is larger than mu
        :return: numpy 1D array: boolian type array (True: is core point)
        """
        is_core_point = np.array([(len(pwn)>=self.mu) & (spd<=self.lamb)
                                  for pwn, spd in zip(self.preference_weighted_neighbourhoods,
                                                      self.subspace_preference_dimensionalities)])
        return is_core_point

    def fit(self):
        """Run clustering algorithm
        :return: cluster labels for each point
        """
        labels_dict = {}          # Stores the processed data points and respective label info
        current_label_id = -1     # Cluster label id
        for point_idx in range(self.nb_points):
            if point_idx not in labels_dict:  # if point has not already been visited
                if self.preference_weighted_core_points[point_idx]:  # Is it a core point?
                    current_label_id += 1  # new cluster -> increment counter
                    # Go through the neighbors and search there fore other cluster members
                    cluster_member_search_list = self.preference_weighted_neighbourhoods[point_idx].tolist()
                    while cluster_member_search_list:
                        cluster_candidate = cluster_member_search_list.pop()
                        possible_cluster_candidates = {cluster_candidate}
                        if self.preference_weighted_core_points[cluster_candidate]:  # If candidate is a core point
                            # go through each point in the neighborhood of the cluster_candidate
                            for candidate_neighbors in self.preference_weighted_neighbourhoods[cluster_candidate]:
                                has_pdim = self.subspace_preference_dimensionalities[cluster_candidate]<=self.lamb
                                is_unclassified = candidate_neighbors not in labels_dict
                                if has_pdim and is_unclassified:
                                    possible_cluster_candidates.add(candidate_neighbors)

                        for new_point in possible_cluster_candidates:
                            if new_point not in labels_dict:
                                if new_point != cluster_candidate:   # should not add the same point again
                                    cluster_member_search_list.append(new_point)
                                labels_dict[new_point] = current_label_id  # Point gets corresponding cluster id
                            elif labels_dict[new_point] == -1:  # if the point is currently classified as noise
                                labels_dict[new_point] = current_label_id

                else:
                    labels_dict[point_idx] = -1  # Save as noise point
        # fill the labels_ array
        for p_idx, label_id in labels_dict.items():
            self.labels_[p_idx] = label_id

        return