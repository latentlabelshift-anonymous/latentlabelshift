import numpy as np
from sklearn.decomposition import NMF

from clustering import *

class ClusterModelSklearnNMF(ClusterModel):
    def __init__(self, base_cluster_model, n_discretization):
        self.base_cluster_model = base_cluster_model
        self.n_discretization = n_discretization

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'base_params': self.base_cluster_model.get_hyperparameter_dict(),
            'n_discretization': self.n_discretization
        }

    def get_p_y_given_discrete_x_d(self, d, p_discrete_x_given_y, p_y_given_domain):
        unscaled = p_discrete_x_given_y.T * np.expand_dims(p_y_given_domain[:,d], axis=1)
        for col in range(unscaled.shape[1]):
            if np.sum(unscaled[:,col], axis=0) > 0:
                unscaled[:,col] /= np.sum(unscaled[:,col], axis=0)
        return unscaled

    def train_cluster(self, n_classes, n_domains, input_data, input_domains):
        # First fit the base cluster model on input_data to discretize
        
        self.base_cluster_model.train_cluster(self.n_discretization, n_domains, input_data, input_domains)
        
        cluster_labels = self.base_cluster_model.eval_cluster(input_data, input_domains)

        discrete_x = cluster_labels
        n_discrete_x = self.n_discretization
        n_latents = n_classes

        bin_edges = [ np.array(list(range(n_discrete_x + 1))) - 0.5, np.array(list(range(n_domains + 1))) - 0.5 ]
        print(discrete_x.shape)
        x_matrix = np.histogram2d(discrete_x, input_domains, bins=bin_edges)[0]

        x_matrix = x_matrix / np.sum(x_matrix, axis=0, keepdims=True)


        from sklearn.decomposition import NMF
        self.model = NMF(n_components=n_latents, init='random')
        
        W_new = self.model.fit_transform(x_matrix.T).T
        
        C_new = self.model.components_.T


        col_sums_C = np.sum(C_new, axis=0, keepdims=True)
        col_sums_C_vec = col_sums_C[0]

        C_new /= col_sums_C
        W_new *= np.tile(np.expand_dims(col_sums_C_vec, axis=1), (1,n_domains))

        W_new /= np.sum(W_new, axis=0, keepdims=True)

        self.p_y_given_d = W_new
        self.p_discrete_x_given_y = C_new


        col_sums_C = np.sum(C_new, axis=0, keepdims=True)
        col_sums_C_vec = col_sums_C[0]



        self.p_discrete_x_given_d = self.p_discrete_x_given_y @ self.p_y_given_d
        self.p_y_given_discrete_x_d = np.stack([self.get_p_y_given_discrete_x_d(d, self.p_discrete_x_given_y, self.p_y_given_d) for d in range(n_domains)], axis=2)
        # dim: 0 - y
        # dim: 1 - x
        # dim: 2 - domains

        predicted_labels = []
        for x, d in zip(discrete_x, input_domains):
            if (x == -1):
                predicted_labels.append(x)
            else:
                label_probs = self.p_y_given_discrete_x_d[:, x, d]
                predicted_labels.append(np.argmax(label_probs, axis=0))
        
        return np.array(predicted_labels).flatten()

    def eval_cluster(self, input_data, input_domains):
        cluster_labels = self.base_cluster_model.eval_cluster(input_data, input_domains)
        discrete_x = cluster_labels

        predicted_labels = []
        for x, d in zip(discrete_x, input_domains):

            if (x == -1):
                predicted_labels.append(x)
            else:
                label_probs = self.p_y_given_discrete_x_d[:, x, d]
                predicted_labels.append(np.argmax(label_probs, axis=0))
        
        return np.array(predicted_labels).flatten()


