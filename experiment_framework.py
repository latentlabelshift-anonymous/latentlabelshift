

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from class_prior import *
from clustering_topic_modeling import *
from clustering import *
from dataset import *
from experiment_utils import *
from feature_extractor import *
from domain_discriminator import *
from permutation_solver import *
from domain_discriminator_scan import *

class ExperimentSetup:
    def __init__(self, dataset, domain_class_prior_matrix, feature_extractor, clusterer, permutation_solver, device):
        self.dataset = dataset
        self.domain_class_prior_matrix = domain_class_prior_matrix
        self.feature_extractor = feature_extractor
        self.clusterer = clusterer
        self.permutation_solver = permutation_solver
        self.n_domains = self.domain_class_prior_matrix.n_domains
        self.n_classes = self.dataset.n_classes

        class_domain_assignment_matrix_train = self.domain_class_prior_matrix.class_domain_assignment_matrix_train
        class_domain_assignment_matrix_test  = self.domain_class_prior_matrix.class_domain_assignment_matrix_test
        class_domain_assignment_matrix_valid = self.domain_class_prior_matrix.class_domain_assignment_matrix_valid

        data_dims = self.dataset.data_dims

        # from here below train_data, test_data, valid_data are all LOADERS

        self.train_data, self.test_data, self.valid_data, self.train_labels, self.test_labels, self.valid_labels = self.format_data(
            class_domain_assignment_matrix_train,
            class_domain_assignment_matrix_test,
            class_domain_assignment_matrix_valid,
            data_dims
        )

        # self.train_data = self.train_data.to(device)
        # self.test_data  = self.test_data.to(device)
        # self.valid_data = self.valid_data.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_labels  = self.test_labels.to(device)
        self.valid_labels = self.valid_labels.to(device)

        train_domains = self.train_labels[:,1]
        valid_domains = self.valid_labels[:,1]
        test_domains = self.test_labels[:,1]

        train_labels_only = self.train_labels[:,0]
        valid_labels_only = self.valid_labels[:,0]
        test_labels_only  = self.test_labels[:,0]

        


        self.feature_extractor.fit_extractor(self.train_data, self.valid_data, self.test_data, train_domains, valid_domains, train_labels_only, valid_labels_only, test_labels_only)

        # Valid for clustering
        cluster_features_train = self.feature_extractor.get_features(
           self.train_data
        )
        cluster_features_valid = self.feature_extractor.get_features(
           self.valid_data
        )
        cluster_features_valid_train = np.concatenate([cluster_features_valid, cluster_features_train], axis=0)
        cluster_features_test = self.feature_extractor.get_features(
            self.test_data
        )


        self.true_test_classes = self.test_labels[:,0].cpu().numpy()

        valid_domains = valid_domains.cpu().numpy()
        train_domains = train_domains.cpu().numpy()
        test_domains = test_domains.cpu().numpy()

        domains_valid_train = np.concatenate([valid_domains, train_domains], axis=0)

        # CLUSTER ON VALID ALONE AND RECORD

        clusterer.train_cluster(
            n_classes=self.n_classes,
            n_domains=self.n_domains,
            input_data=cluster_features_valid,
            input_domains=valid_domains)
        clustered_test_labels = clusterer.eval_cluster(cluster_features_test, input_domains=test_domains)

        # self.handle_clustered_test_labels(clustered_test_labels)
        permuted_labels = self.permutation_solver.get_best_permutation(clustered_test_labels, self.true_test_classes)
        test_accuracy = label_accuracy(permuted_labels, self.true_test_classes)
        self.permuted_labels = permuted_labels
        self.test_accuracy = test_accuracy

        # CLUSTER ON TRAIN + VALID

        clusterer.train_cluster(
            n_classes=self.n_classes,
            n_domains=self.n_domains,
            input_data=cluster_features_valid_train,
            input_domains=domains_valid_train)
        clustered_test_labels = clusterer.eval_cluster(cluster_features_test, input_domains=test_domains)

        # self.handle_clustered_test_labels(clustered_test_labels)
        permuted_labels = self.permutation_solver.get_best_permutation(clustered_test_labels, self.true_test_classes)
        self.test_accuracy_valid_train = label_accuracy(permuted_labels, self.true_test_classes)


        # OTHER METRICS

        predicted_class_prior = np.zeros_like(self.clusterer.p_y_given_d)
        best_class_ordering = self.permutation_solver.best_class_ordering                
        for class_label, new_class_label in enumerate(best_class_ordering):
            predicted_class_prior[new_class_label,:] = self.clusterer.p_y_given_d[class_label,:]

        reconstruction_error_L1 = np.sum(abs(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior))
        self.test_post_cluster_p_y_given_d_l1_norm = reconstruction_error_L1

        reconstruction_error = np.linalg.norm(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior)
        self.test_post_cluster_p_y_given_d_fro_norm = reconstruction_error

        # uniform
        p_d_x = cluster_features_test#.detach().cpu().numpy()
        p_y_d = clusterer.p_y_given_d#.detach().cpu().numpy()
        solved_dd_test_labels, p_y_x = y_predictions_dd_uniform(p_d_x, p_y_d)

        permuted_labels = self.permutation_solver.get_best_permutation(solved_dd_test_labels, self.true_test_classes)
        self.test_post_cluster_acc_dd_uniform = label_accuracy(permuted_labels, self.true_test_classes)

        predicted_class_prior = np.zeros_like(self.clusterer.p_y_given_d)
        best_class_ordering = self.permutation_solver.best_class_ordering                
        for class_label, new_class_label in enumerate(best_class_ordering):
            predicted_class_prior[new_class_label,:] = self.clusterer.p_y_given_d[class_label,:]

        reconstruction_error_L1_uniform = np.sum(abs(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior))
        self.test_post_cluster_p_y_given_d_l1_norm_uniform = reconstruction_error_L1_uniform

        reconstruction_error_uniform = np.linalg.norm(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior)
        self.test_post_cluster_p_y_given_d_fro_norm_uniform = reconstruction_error_uniform

        # domain adjusted
        solved_dd_test_labels_balanced, _ = y_predictions_dd_balanced(p_y_d, p_y_x, test_domains)
        permuted_labels = self.permutation_solver.get_best_permutation(solved_dd_test_labels_balanced, self.true_test_classes)
        self.test_post_cluster_acc_dd_balanced = label_accuracy(permuted_labels, self.true_test_classes)

        predicted_class_prior = np.zeros_like(self.clusterer.p_y_given_d)
        best_class_ordering = self.permutation_solver.best_class_ordering                
        for class_label, new_class_label in enumerate(best_class_ordering):
            predicted_class_prior[new_class_label,:] = self.clusterer.p_y_given_d[class_label,:]

        reconstruction_error_L1_balanced = np.sum(abs(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior))
        self.test_post_cluster_p_y_given_d_l1_norm_balanced = reconstruction_error_L1_balanced

        reconstruction_error_balanced = np.linalg.norm(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior)
        self.test_post_cluster_p_y_given_d_fro_norm_balanced = reconstruction_error_balanced


    def format_data(self, class_domain_assignment_matrix_train, class_domain_assignment_matrix_test, class_domain_assignment_matrix_valid, data_dims):
        train_data, train_labels = self.build_data_label_matrix(self.dataset.train_data, self.dataset.train_label_concatenate, class_domain_assignment_matrix_train, self.dataset.n_train, self.domain_class_prior_matrix.n_domains, data_dims)
        test_data, test_labels = self.build_data_label_matrix(self.dataset.test_data, self.dataset.test_label_concatenate, class_domain_assignment_matrix_test, self.dataset.n_test, self.domain_class_prior_matrix.n_domains, data_dims)
        valid_data, valid_labels = self.build_data_label_matrix(self.dataset.valid_data, self.dataset.valid_label_concatenate, class_domain_assignment_matrix_valid, self.dataset.n_valid, self.domain_class_prior_matrix.n_domains, data_dims)
        return train_data, test_data, valid_data, train_labels, test_labels, valid_labels

    def build_data_label_matrix(self, dataset, label_concatenate, class_domain_assignment_matrix, n_data, n_domains, data_dims):
        dims = [int(torch.sum(class_domain_assignment_matrix))] + list(data_dims)
        data = torch.zeros(*dims)
        labels = torch.zeros(int(torch.sum(class_domain_assignment_matrix)) , 2).long()
        done_flag = False
        data_index = 0

        indices_list = []

        copy_class_domain_assignment_matrix = class_domain_assignment_matrix.clone()
        for i in range(int(n_data)):
            if done_flag:
                break
            class_label = label_concatenate[i, 0]
            for domain in range(n_domains):
                still_needed = copy_class_domain_assignment_matrix[domain, int(class_label)]
                if still_needed > 0:
                    copy_class_domain_assignment_matrix[domain, int(class_label)] -= 1
                    label_concatenate[i, 1] = domain

                    # data[data_index] = data_concatenate[i]
                    indices_list.append(i)
                    labels[data_index]   = label_concatenate[i]
                    
                    if data_index == data.shape[0]:
                        done_flag = True

                    data_index += 1
                    break

        if labels.shape[0] > len(indices_list):
            labels = labels[:len(indices_list)]
        
        indices_list, labels = shuffle(indices_list, labels)

        dataset_subset = torch.utils.data.Subset(dataset, indices_list)
        data = torch.utils.data.DataLoader(dataset_subset, batch_size=32, shuffle=False)

        # return a loader

        return data, labels
