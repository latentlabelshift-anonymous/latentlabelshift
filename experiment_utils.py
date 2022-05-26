import numpy as np
# from scipy.linalg import solve
import torch

def label_accuracy(label_output, true_classes):
    return 1 - np.count_nonzero(label_output - true_classes) / len(label_output)

def get_name(object):
    return object.__class__.__name__

coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

def y_predictions_dd_uniform(p_d_x, p_y_d):
    p_y_x = solve_p_y_x(p_d_x, p_y_d)
    return np.argmax(p_y_x, axis=1), p_y_x

def y_predictions_dd_balanced(p_y_d, p_y_x, true_domains):
    p_y_x_d = solve_p_y_x_d(p_y_d, p_y_x, true_domains)
    return np.argmax(p_y_x_d, axis=1), p_y_x_d

def solve_p_y_x(p_d_x, p_y_d):
    # note p_d_x batched
    p_d_x_t = p_d_x.T
    p_d_y = p_y_d.T / np.where(np.sum(p_y_d.T, axis=0, keepdims=True) == 0, 1e-8, np.sum(p_y_d.T, axis=0, keepdims=True))
    p_y_x_t = np.linalg.pinv(p_d_y) @ p_d_x_t
    p_y_x = p_y_x_t.T
    return p_y_x

def solve_p_y_x_d(p_y_d, p_y_x, true_domains):

    p_d_y = p_y_d.T / np.where(np.sum(p_y_d.T, axis=0, keepdims=True) == 0, 1e-8, np.sum(p_y_d.T, axis=0, keepdims=True))

    p_d_y_list = [
        p_d_y[domain:domain+1, :]
        for domain in true_domains
    ]
    # each row is p(d_true | y)
    p_d_y_rows = np.concatenate(p_d_y_list, axis=0)

    p_y_x_d = p_y_x * p_d_y_rows
    p_y_x_d /= np.sum(p_y_x_d, axis=1, keepdims=True) # normalize rows to account for denominator
    return p_y_x_d


def model_evaluate(model,dataset_loader, dataset_labels,ps, device): 
    batch_size=  32

    test_labels = []

    model.eval()
    output_list = []
    for vec in dataset_loader:
        if isinstance(vec, dict):
            batch, labels = vec['image'], vec['target']
        else:
            batch, labels = vec
        batch = batch.to(device)
        with torch.no_grad(): 
            print(batch.shape)
            cluster_softmax = model(batch)[0] 
            _ , cluster_assignments = torch.max(cluster_softmax,dim=1)
            output_list.append(cluster_assignments.cpu().numpy()) 
        test_labels.append(labels.cpu().numpy())
    
    prediction_outputs = np.concatenate(output_list,axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    permuted_labels = ps.get_best_permutation(prediction_outputs , test_labels) 

    return label_accuracy(permuted_labels,test_labels) 

