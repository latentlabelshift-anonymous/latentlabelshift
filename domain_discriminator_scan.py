import copy
import scipy
import torch
from torch import dropout, nn
import torch.nn.functional as F

import numpy as np
import wandb
from models.resnet import  *

from outside_model_definitions import *

from feature_extractor import *

import torchvision


class DomainDiscriminatorSCAN(FeatureExtractor):
    
    def __init__(self, device, lr, exp_lr_gamma, epochs, batch_size, n_classes, n_domains,load_path=None,eval_clusterer=None, eval_ps=None, class_prior=None, epoch_interval_to_compute_final_task=100,dropout=0,limit_gradient_flow=False,use_scheduler='ExponentialLR',baseline_load_path=None):
        
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.load_path = load_path
        
        try:
            assert(dropout in [0,1])
        except: 
            dropout = 1

        self.dropout = dropout
        self.limit_gradient_flow = limit_gradient_flow

        self.model = self.build_model().to(device)
        self.epochs = epochs
        self.lr = lr
        self.gamma = exp_lr_gamma
        self.device = device
        self.batch_size = batch_size

        self.epoch_interval_to_compute_final_task = epoch_interval_to_compute_final_task
        self.eval_clusterer = eval_clusterer
        self.eval_ps = eval_ps
        self.class_prior = class_prior
        assert(use_scheduler in ['ReduceLROnPlateau' , 'ExponentialLR'])
        self.use_scheduler = use_scheduler

        self.baseline_load_path = baseline_load_path

    def build_model(self):
        pass

    def baseline_acc(self, test_data, test_labels):
        pass

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_domains': self.n_domains,
            'n_epochs': self.epochs,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'final_task_epoch_interval': self.epoch_interval_to_compute_final_task,
            'eval_clusterer': self.eval_clusterer.get_hyperparameter_dict(),
            'dropout' : self.dropout,
            'limit_gradient_flow' : self.limit_gradient_flow,
            'use_scheduler' : self.use_scheduler,
            'load_path': self.load_path,
            'baseline_load_path': self.baseline_load_path
        }

    def fit_extractor(self, train_data, valid_data, test_data, train_domains, valid_domains , train_labels ,valid_labels, test_labels):

        # Pre-compute best possible acc
        scan_alone_test_acc = self.baseline_acc(test_data, test_labels)


        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if(self.use_scheduler == 'ExponentialLR'):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        elif(self.use_scheduler == 'ReduceLROnPlateau'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.01,patience=5)



        batch_size = self.batch_size

        # compute bayes optimal accuracy (depecrated)

        self.bayes_optimal_dd_acc = 1


        self.model.train()

        valid_post_cluster_acc = 0
        valid_post_cluster_p_y_given_d_fro_norm = 0
        valid_post_cluster_acc_dd_uniform = 0
        valid_post_cluster_acc_dd_balanced = 0
        valid_post_cluster_p_y_given_d_l1_norm = 0

        valid_post_cluster_p_y_given_d_fro_norm_uniform = 0
        valid_post_cluster_p_y_given_d_l1_norm_uniform  = 0

        valid_post_cluster_p_y_given_d_fro_norm_balanced  = 0
        valid_post_cluster_p_y_given_d_l1_norm_balanced  = 0

        best_epoch = 0
        best_model = None
        best_valid_loss = None

        for epoch in range(self.epochs):
            self.model.train()

            n_correct = 0
            sum_loss = 0
            n_train = 0
            batch_start = 0
            for vec in train_data:
                if isinstance(vec, dict):
                    batch, _ = vec['image'], vec['target']
                else:
                    batch, _ = vec
                # batch = train_data[batch_start:batch_start + batch_size]
                batch = batch.to(self.device)
                labels = train_domains[batch_start:batch_start + batch_size]

                optimizer.zero_grad()
                logits = self.model(batch)
                loss = loss_fn(logits, labels)
                
                sum_loss += loss.detach().cpu().numpy()
                loss.backward()
                optimizer.step()

                n_correct_batch = len(torch.nonzero(torch.argmax(logits, dim=1) == labels))
                n_correct += n_correct_batch
                n_train += batch.shape[0]
                batch_start += batch.shape[0]
            train_acc = n_correct / n_train
            train_loss = sum_loss / n_train

            self.model.eval()

            n_correct = 0
            sum_loss = 0
            n_valid = 0
            batch_start = 0


            for vec in valid_data:
                if isinstance(vec, dict):
                    batch, _ = vec['image'], vec['target']
                else:
                    batch, _ = vec
                # batch = valid_data[batch_start:batch_start + batch_size]
                batch = batch.to(self.device)
                labels = valid_domains[batch_start:batch_start + batch_size]

                with torch.no_grad():

                    logits = self.model(batch)
                    loss = loss_fn(logits, labels)
                    
                    sum_loss += loss.detach().cpu().numpy()

                n_correct_batch = len(torch.nonzero(torch.argmax(logits, dim=1) == labels))

                n_correct += n_correct_batch
                n_valid += batch.shape[0]
                batch_start += batch.shape[0]
            valid_acc = n_correct / n_valid
            valid_loss = sum_loss / n_valid

            if(self.use_scheduler == 'ExponentialLR'):
                scheduler.step()
            elif(self.use_scheduler == 'ReduceLROnPlateau'):
                scheduler.step(valid_loss) 

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                best_valid_loss = valid_loss
            
            if epoch % self.epoch_interval_to_compute_final_task == 0:
                # compute post cluster acc
                curr_valid_features = self.get_features(valid_data)

                self.eval_clusterer.train_cluster(
                    n_classes=self.n_classes,
                    n_domains=self.n_domains,
                    input_data=curr_valid_features,
                    input_domains=valid_domains.cpu().numpy())
                clustered_valid_labels = self.eval_clusterer.eval_cluster(curr_valid_features,  input_domains=valid_domains.cpu().numpy())
                
                # handle cluster output
                permuted_labels = self.eval_ps.get_best_permutation(clustered_valid_labels, valid_labels.cpu().numpy())
                valid_post_cluster_acc = label_accuracy(permuted_labels, valid_labels.cpu().numpy())

                predicted_class_prior = np.zeros_like(self.eval_clusterer.p_y_given_d)
                best_class_ordering = self.eval_ps.best_class_ordering                
                for class_label, new_class_label in enumerate(best_class_ordering):
                    predicted_class_prior[new_class_label,:] = self.eval_clusterer.p_y_given_d[class_label,:]

                reconstruction_error = np.linalg.norm(self.class_prior.class_priors.T - predicted_class_prior)
                valid_post_cluster_p_y_given_d_fro_norm = reconstruction_error
                reconstruction_error_L1 = np.sum(abs(self.class_prior.class_priors.T - predicted_class_prior))
                valid_post_cluster_p_y_given_d_l1_norm = reconstruction_error_L1
                valid_frac_clustered = np.count_nonzero(clustered_valid_labels != -1) / len(clustered_valid_labels)

                # also compute valid labels in a different way (using predicted p(d|x))

                # uniform
                p_d_x = curr_valid_features
                p_y_d = self.eval_clusterer.p_y_given_d
                solved_dd_valid_labels, p_y_x = y_predictions_dd_uniform(p_d_x, p_y_d)

                permuted_labels = self.eval_ps.get_best_permutation(solved_dd_valid_labels, valid_labels.cpu().numpy())
                valid_post_cluster_acc_dd_uniform = label_accuracy(permuted_labels, valid_labels.cpu().numpy())

                predicted_class_prior = np.zeros_like(self.eval_clusterer.p_y_given_d)
                best_class_ordering = self.eval_ps.best_class_ordering                
                for class_label, new_class_label in enumerate(best_class_ordering):
                    predicted_class_prior[new_class_label,:] = self.eval_clusterer.p_y_given_d[class_label,:]

                reconstruction_error_uniform = np.linalg.norm(self.class_prior.class_priors.T - predicted_class_prior)
                valid_post_cluster_p_y_given_d_fro_norm_uniform = reconstruction_error_uniform
                reconstruction_error_L1_uniform = np.sum(abs(self.class_prior.class_priors.T - predicted_class_prior))
                valid_post_cluster_p_y_given_d_l1_norm_uniform = reconstruction_error_L1_uniform

                # domain adjusted
                solved_dd_valid_labels_balanced, _ = y_predictions_dd_balanced(p_y_d, p_y_x, valid_domains)
                permuted_labels = self.eval_ps.get_best_permutation(solved_dd_valid_labels_balanced, valid_labels.cpu().numpy())
                valid_post_cluster_acc_dd_balanced = label_accuracy(permuted_labels, valid_labels.cpu().numpy())

                predicted_class_prior = np.zeros_like(self.eval_clusterer.p_y_given_d)
                best_class_ordering = self.eval_ps.best_class_ordering                
                for class_label, new_class_label in enumerate(best_class_ordering):
                    predicted_class_prior[new_class_label,:] = self.eval_clusterer.p_y_given_d[class_label,:]

                reconstruction_error_balanced = np.linalg.norm(self.class_prior.class_priors.T - predicted_class_prior)
                valid_post_cluster_p_y_given_d_fro_norm_balanced = reconstruction_error_balanced
                reconstruction_error_L1_balanced = np.sum(abs(self.class_prior.class_priors.T - predicted_class_prior))
                valid_post_cluster_p_y_given_d_l1_norm_balanced = reconstruction_error_L1_balanced

            if (epoch > 0 and epoch % self.epoch_interval_to_compute_final_task == 0) or (epoch == self.epochs - 1):
                model_save_path = f'{epoch}_{self.n_classes}_{self.n_domains}_{get_name(self)}_{self.class_prior.class_prior_alpha}.pth'
                torch.save(self.model.state_dict(), model_save_path)
                wandb.save(model_save_path)

            wandb.log({
                'train_domain_discriminator_accuracy': train_acc,
                'train_domain_discriminator_loss':     train_loss,
                'valid_domain_discriminator_accuracy': valid_acc,
                'valid_domain_discriminator_loss':    valid_loss,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'valid_final_accuracy': valid_post_cluster_acc,
                'bayes_optimal_valid_domain_discriminator_accuracy': self.bayes_optimal_dd_acc,
                'valid_domain_discriminator_accuracy_ratio': valid_acc / self.bayes_optimal_dd_acc,
                'valid_frac_clustered': valid_frac_clustered,
                'valid_final_accuracy_normalized_frac_clustered': valid_post_cluster_acc / valid_frac_clustered,
                'valid_post_cluster_acc_dd_uniform': valid_post_cluster_acc_dd_uniform,
                'valid_post_cluster_acc_dd_balanced': valid_post_cluster_acc_dd_balanced,

                'valid_post_cluster_p_y_given_d_fro_norm': valid_post_cluster_p_y_given_d_fro_norm,
                'valid_post_cluster_p_y_given_d_l1_norm': valid_post_cluster_p_y_given_d_l1_norm,

                'valid_post_cluster_p_y_given_d_fro_norm_uniform': valid_post_cluster_p_y_given_d_fro_norm_uniform,
                'valid_post_cluster_p_y_given_d_l1_norm_uniform': valid_post_cluster_p_y_given_d_l1_norm_uniform,

                'valid_post_cluster_p_y_given_d_fro_norm_balanced': valid_post_cluster_p_y_given_d_fro_norm_balanced,
                'valid_post_cluster_p_y_given_d_l1_norm_balanced': valid_post_cluster_p_y_given_d_l1_norm_balanced,
                'scan_alone_test_acc': scan_alone_test_acc
            })



        # Preserve best model on test dataset
        self.model = best_model

        model_save_path = f'best_{best_epoch}_{self.n_classes}_{self.n_domains}_{get_name(self)}_{self.class_prior.class_prior_alpha}.pth'
        torch.save(self.model.state_dict(), model_save_path)
        wandb.save(model_save_path)


    def get_features(self, data):
        # batch_size = 32
        eval_probs_list =[]
        softmax = nn.Softmax().to(self.device)
        self.model.eval()
        corr_labels = []
        for vec in data:
            if isinstance(vec, dict):
                batch, _ = vec['image'], vec['target']
            else:
                batch, _ = vec
            corr_labels.append(_)
            batch = batch.to(self.device)
            # batch = data[i:i+batch_size]
            with torch.no_grad():
                eval_logits = self.model(batch)
                eval_probs = softmax(eval_logits).cpu().numpy()
            eval_probs_list.append(eval_probs)

        self.corr_labels = np.concatenate(corr_labels, axis=0)

        return np.concatenate(eval_probs_list, axis=0)


class scan_pretext(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels):
        load_path = self.baseline_load_path
        backbone = resnet18()
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model'] 
        model.load_state_dict(state_dict,strict=True)
        scan_alone_test_acc = model_evaluate(model, test_data, test_labels, self.eval_ps, self.device)

        return scan_alone_test_acc

    def build_model(self):
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow
        if(dropout == 0):
            obj =  ResNetwithInitialisations(BasicBlock,[2,2,2,2],num_classes=num_classes) 
        else: 
            obj = ResNetwithInitialisations(BasicBlockWithDropout,[2,2,2,2],num_classes=num_classes)
            
        load_path = self.load_path
        state = torch.load(load_path)
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state.keys(): 
            renamed_state[key.replace('backbone.','')] = state[key] 
        
        obj.load_state_dict(renamed_state, strict=False)

        if(limit_gradient_flow == True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 

        return obj

class scan_scan(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels):
        load_path = self.baseline_load_path
        backbone = resnet18()
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model'] 
        model.load_state_dict(state_dict,strict=True)
        scan_alone_test_acc = model_evaluate(model, test_data, test_labels, self.eval_ps, self.device)

        return scan_alone_test_acc

    def build_model(self): 
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow
        if(dropout == 0):
            obj =  ResNetwithInitialisations(BasicBlock,[2,2,2,2],num_classes=num_classes) 
        else: 
            obj = ResNetwithInitialisations(BasicBlockWithDropout,[2,2,2,2],num_classes=num_classes)
         
        load_path = self.load_path
        state_full = torch.load(load_path)
        state_dict = state_full['model']
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state_dict.keys(): 
            renamed_state[key.replace('backbone.','')] = state_dict[key]

        obj.load_state_dict(renamed_state,strict=False)

        if(limit_gradient_flow==True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 
        return obj 

class scan_scan_imagenet(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels):
        
        load_path = self.baseline_load_path
        obj  = torchvision.models.__dict__['resnet50']()
        obj.fc = nn.Identity()
        backbone = {'backbone': obj, 'dim':2048}
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model'] 
        missing = model.load_state_dict(state_dict,strict=False)
        scan_alone_test_acc = model_evaluate(model, test_data, test_labels, self.eval_ps, self.device)

        return scan_alone_test_acc

    def build_model(self): 
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow

        if(dropout != 0):
            print("Provision to add dropout has not yet been implemented") 
            raise ValueError 

        obj =  torchvision.models.__dict__['resnet50']()
        # obj = torchvision.models.resnet50()
        obj.fc = nn.Linear(2048,num_classes,bias=True) 
         
        load_path = self.load_path
        state_full = torch.load(load_path)
        state_dict = state_full['model']
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state_dict.keys(): 
            renamed_state[key.replace('backbone.','')] = state_dict[key]

        obj.load_state_dict(renamed_state,strict=False)

        if(limit_gradient_flow==True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 
        return obj

class scan_selffinetuned(DomainDiscriminatorSCAN): 
    def build_model(self):
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow

        if(dropout == 0):
            obj =  ResNetwithInitialisations(BasicBlock,[2,2,2,2],num_classes=num_classes) 
        else: 
            obj = ResNetwithInitialisations(BasicBlockWithDropout,[2,2,2,2],num_classes=num_classes)

        load_path = self.load_path 
        state = torch.load(load_path) 
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state.keys(): 
            renamed_state[key.replace('backbone.','')] = state[key] 

        obj.load_state_dict(renamed_state,strict=False) 

        if(limit_gradient_flow == True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 
        return obj 

        




    

        