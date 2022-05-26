from sklearn.decomposition import PCA, FastICA
import torch

from experiment_utils import *

class FeatureExtractor:

    def fit_extractor(self, train_data, valid_data, train_domains, valid_domains, train_labels, valid_labels):
        pass

    def get_features(self, data):
        pass

    def get_hyperparameter_dict(self):
        pass

