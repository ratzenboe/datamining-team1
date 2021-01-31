import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import copy


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

        # Nearest neighbor structure
        self.neigbors_clf = None
            
        # Fill member variables at initialization
        self.epsilon_neighbourhoods = self.create_epsilon_neighbourhoods()
        self.attribute_variances = self.create_variances_along_attributes()
        self.subspace_preference_dimensionalities = self.create_subspace_preference_dimensionality()
        self.subspace_preference_vectors = self.create_subspace_preference_vectors()
        self.pref_weighted_similarity_measures = self.create_preference_weighted_similarity_matrix()
        self.preference_weighted_neighbourhoods = self.create_preference_weighted_epsilon_neighbourhood()
        self.preference_weighted_core_points = self.create_preference_weighted_core_points()

        # Cluster label information
        self.labels_ = np.full(shape=(self.nb_points,), fill_value=np.nan)

    def create_epsilon_neighbourhoods(self):
        """Calculate epsilon-neighbourhood for each data point
        :return: numpy ndarray where each row contains the indices of points that
                 are in the neighborhood of a certain point
        """
        self.neigbors_clf = NearestNeighbors(radius=self.epsilon, algorithm='ball_tree')
        self.neigbors_clf.fit(self.data)
        _, neigh_idx = self.neigbors_clf.radius_neighbors(self.data)
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
        """Calcuate sparse similarity matrix of all points in the data set
        :return: numpy ndarray: symmetrical distance matrix using the weighted similarity measure
        """
        # We only need to compare the distances to points within the epsilon shell (to determine if a point is a core point)
        # Since the subspace scaling factor kappa is >>1 (and not <1), no distances to other points will be needed for 
        # the core point evaluation

        # get points in epsilon shell: attententio point itself is not in neigh_ind list
        _, neigh_ind = self.neigbors_clf.radius_neighbors(radius=self.epsilon)
        row, col, pwsim = [], [], []
        for i, ith_neigh_ind in enumerate(neigh_ind):
            # Calculate preference weighted similarity measure with point and neighbors in eps shell
            sq_diffs = np.square(self.data[ith_neigh_ind,:] - self.data[i,:])
            sum_weighted_sq_diffs = np.inner(self.subspace_preference_vectors[i,:], sq_diffs)
            pwsim_ith = np.sqrt(sum_weighted_sq_diffs)
            
            # Info for sparse matrix
            pwsim.extend(pwsim_ith.tolist())      # Data
            row.extend([i]*(pwsim_ith.shape[0]))  # ith Row 
            col.extend(ith_neigh_ind.tolist())    # column info

        # Construct sparse matrix with data, row, and column info
        A = csr_matrix((pwsim, (row, col)), shape=(self.nb_points, self.nb_points))
        # Create symmetric version: take the elementwise maximum of A and its transpose A.T
        transpose_is_bigger = A.T>A
        A = A - A.multiply(transpose_is_bigger) + (A.T).multiply(transpose_is_bigger)
        
        return A
    

    def create_preference_weighted_epsilon_neighbourhood(self):
        """New neighborhood definition under the weighted similarity measure
        :return: List of numpy 1D arrays containing indices of points in the preference weighted eps. nbh
        """
        
        A = self.pref_weighted_similarity_measures   # distances matrix
        A[A>self.epsilon] = 0   # set distances greater than epsilon to 0
        A.eliminate_zeros()     # then remove these entries from matrix
        # For each entry in data get neighbor indices with preference weighted distance less than epsilon
        weighted_eps_neighbh = np.split(A.indices, A.indptr)[1:-1]  

        return weighted_eps_neighbh

    def create_preference_weighted_core_points(self):
        """ For each point in the data set test if the conditions 1&2 to a preference weighted core point are satisfied:
            1. Subspace dimensionality is smaller or equal than lambda
            2. The number of preference weighted epsilon neighbors is larger than mu
        :return: numpy 1D array: boolian type array (True: is core point)
        """
        is_core_point = np.array([(len(pwn)+1>=self.mu) & (spd<=self.lamb)   # +1 because point itself is not taken into account in radius neighbor query
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
                    # Go through the neighbors and search there for other cluster members
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