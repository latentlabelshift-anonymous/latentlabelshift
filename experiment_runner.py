import os
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import argparse
parser = argparse.ArgumentParser(description='Choose dataset, seed wave')
parser.add_argument('--dataset', default=None, type=str,
                     help='choose dataset from [cifar10, cifar20, imagenet, fg2, fg28]')

parser.add_argument('--random_seed_wave', type=int, default=1,
                     help='choose set of random seeds from [1,2,3,4]')
parser.add_argument('--GPU', type=int, default=0, help='choose GPU')
parser.add_argument('--start_alpha_index', type=int, default=0, help='choose start index of alpha list from [0,1,2]')
parser.add_argument('--start_domain_index', type=int, default=0, help='choose start index of domain list (options vary by dataset)')
args = parser.parse_args()

# IMPORTANT: DO THIS BEFORE ANY OTHER IMPORTS. Must be set before other imports loaded
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

from dataset import *
from permutation_solver import *
from experiment_utils import *
from outside_model_definitions import *
from domain_discriminator_scan import * 



dataset_choice = args.dataset
random_seed_wave = args.random_seed_wave


if random_seed_wave == 1:
    class_prior_seeds   = [4, 23, 623, 23423, 66]
    dataseed_seeds      = [7636,25,236,123,823]
elif random_seed_wave == 2:
    class_prior_seeds   = [268773, 9947296, 76383, 2234, 12839]
    dataseed_seeds      = [938759, 2827476, 9283896, 776768, 2658462]
elif random_seed_wave == 3:
    class_prior_seeds   = [78496, 692, 7766739, 1112, 951]
    dataseed_seeds      = [112684, 16573, 56396, 309909, 9234]
elif random_seed_wave == 4:
    class_prior_seeds   = [89234, 8847675, 123456786, 93939, 659256]
    dataseed_seeds      = [999, 395, 1001, 500700, 99029503]

import _pickle as cPickle
import wandb

from experiment_framework import *

if dataset_choice == 'cifar10':
    domain_range = [10,15,20,25] #CIFAR10
    max_cond_numbers_cifar10 = [4,4,8]
    max_cond_numbers = max_cond_numbers_cifar10

    dummy_dataset_instance = CIFAR10(42)

    dataset_class = CIFAR10

    scan_ddfa_epochs        = 25
    scan_ddfa_loadpath      = 'scan_cifar-10.pth.tar'
    scan_ddfa_subclass_name = scan_scan

    baseline_scan_name      = scan_ddfa_loadpath

    ddfa_epochs             = 100
    ddfa_n_discretization   = 30

    use_raw_ddfa = True

elif dataset_choice == 'cifar20':
    domain_range = [20,25,30] #CIFAR20
    max_cond_numbers_cifar20 = [8,12,20]
    max_cond_numbers = max_cond_numbers_cifar20

    dummy_dataset_instance = CIFAR20(42)

    dataset_class = CIFAR20

    scan_ddfa_epochs        = 25
    scan_ddfa_loadpath      = 'scan_cifar-20.pth.tar'
    scan_ddfa_subclass_name = scan_scan

    baseline_scan_name      = scan_ddfa_loadpath

    ddfa_epochs             = 100
    ddfa_n_discretization   = 60

    use_raw_ddfa = True

elif dataset_choice == 'imagenet':
    domain_range = [50, 60] #Imagenet
    max_cond_numbers_Imagenet = [200,205,210]
    max_cond_numbers = max_cond_numbers_Imagenet

    dummy_dataset_instance = ImageNet50(42)

    dataset_class = ImageNet50

    scan_ddfa_epochs        = 25
    scan_ddfa_loadpath      = 'scan_imagenet_50.pth.tar'
    scan_ddfa_subclass_name = scan_scan_imagenet

    baseline_scan_name      = scan_ddfa_loadpath


    use_raw_ddfa = False

elif dataset_choice == 'fg2':
    domain_range = [10,7,5,3,2] # fg2
    max_cond_numbers_fg2 = [3,5,7]
    max_cond_numbers = max_cond_numbers_fg2

    dummy_dataset_instance = FieldGuide2(42)

    dataset_class = FieldGuide2

    scan_ddfa_epochs        = 1#30
    scan_ddfa_loadpath      = './scan_fieldguide_run/fieldguide2/pretext/model.pth.tar'
    scan_ddfa_subclass_name = scan_pretext

    # for comparison
    baseline_scan_name      = './scan_fieldguide_run/fieldguide2/scan/model.pth.tar'


    use_raw_ddfa = False

elif dataset_choice == 'fg28':
    domain_range = [47, 42, 37, 28]# fg28
    max_cond_numbers_fg28 = [12, 20, 28]
    max_cond_numbers = max_cond_numbers_fg28

    dummy_dataset_instance = FieldGuide28(42)

    dataset_class = FieldGuide28

    scan_ddfa_epochs        = 60
    scan_ddfa_loadpath      = './scan_fieldguide_run/fieldguide28/pretext/model.pth.tar'
    scan_ddfa_subclass_name = scan_pretext

    # for comparison
    baseline_scan_name      = './scan_fieldguide_run/fieldguide28/scan/model.pth.tar'


    use_raw_ddfa = False
else:
    print(f'dataset {dataset_choice} not recognized')



alphas = [0.5,3,10]

alphas = alphas[args.start_alpha_index:]
max_cond_numbers = max_cond_numbers[args.start_alpha_index:]



for alpha_index, alpha, max_cond_number in zip(range(len(alphas)), alphas, max_cond_numbers):
    runs = []

    setups = list(zip(domain_range, class_prior_seeds, dataseed_seeds))

    if alpha_index == args.start_alpha_index:
        setups = setups[args.start_domain_index:]

    for domains, class_prior_seed, data_seed in setups:
        print(alpha, 'sampling')
        class_prior = RandomDomainClassPriorMatrix(
            n_classes = dummy_dataset_instance.n_classes, 
            n_domains = domains, 
            max_condition_number = max_cond_number, 
            random_seed = class_prior_seed, 
            class_prior_alpha = alpha, 
            min_train_num = dummy_dataset_instance.min_train_num,
            min_test_num = dummy_dataset_instance.min_test_num, 
            min_valid_num = dummy_dataset_instance.min_valid_num
        )
        
        dataset_instance = dataset_class(dataset_seed=data_seed)

        # Add scan main run 
        runs.append({
            'n_domains': domains,
            'class_prior': class_prior,
            'clusterer': ClusterModelSklearnNMF(
                base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),
                n_discretization = dummy_dataset_instance.n_classes,
            ),
            'dataset': dataset_instance,
            'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
            'extractor': scan_ddfa_subclass_name(
                    device,
                    lr = 0.00001,
                    exp_lr_gamma = 0.97,
                    epochs = scan_ddfa_epochs,
                    batch_size = 32,
                    n_classes= class_prior.n_classes,
                    n_domains = domains,
                    load_path= scan_ddfa_loadpath,
                    eval_clusterer = ClusterModelSklearnNMF(
                        base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),
                        n_discretization = dummy_dataset_instance.n_classes,
                    ),
                
                    eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                    class_prior = class_prior,
                    epoch_interval_to_compute_final_task=100,
                    dropout = 0,
                    limit_gradient_flow=False,
                    use_scheduler = 'ExponentialLR',
                    baseline_load_path=baseline_scan_name


            ),
            'alpha': alpha
        })


        if use_raw_ddfa:        
            # Add Domain Discriminator run
            runs.append({
                'n_domains': domains,
                'class_prior': class_prior,
                'clusterer': ClusterModelSklearnNMF(
                    base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),
                    n_discretization = ddfa_n_discretization,
                ),
                'dataset': dataset_instance,
                'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
                'extractor': CIFAR10PytorchCifar(
                # 'extractor': CIFAR10PytorchCifar(
                    device = device,
                    lr = 0.001,
                    exp_lr_gamma = 0.97,

                    epochs = ddfa_epochs,

                    batch_size = 32,
                    n_classes = class_prior.n_classes,
                    n_domains = domains,
                    eval_clusterer = ClusterModelSklearnNMF(

                        base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),

                        n_discretization = ddfa_n_discretization,
                    ),
                    eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                    class_prior = class_prior
                ),
            })

    for r in runs:

        n_domains          = r['n_domains']
        class_prior        = r['class_prior']
        clusterer          = r['clusterer']
        permutation_solver = r['permutation_solver']
        extractor          = r['extractor']
        dataset_instance   = r['dataset']

        config = {
            component_name : component.get_hyperparameter_dict()
            for component_name, component in [
                ('dataset', dataset_instance),
                ('class_prior', class_prior),
                ('clusterer', clusterer),
                ('permutation_solver', permutation_solver),
                ('feature_extractor', extractor)
            ]
        }

        run = wandb.init(
            entity="entity",
            project="project",
            reinit=True,
            config=config
        )

        experiment = ExperimentSetup(dataset_instance, class_prior, extractor, clusterer, permutation_solver, device)

        run.summary['final_best_labels']    = experiment.permuted_labels
        wandb.config.update({"final_best_labels": list(experiment.permuted_labels)})
        wandb.config.update({"final_test_accuracy": experiment.test_accuracy})
        
        wandb.config.update({"final_test_accuracy_valid_train": experiment.test_accuracy_valid_train})



        wandb.config.update({'test_post_cluster_p_y_given_d_l1_norm': experiment.test_post_cluster_p_y_given_d_l1_norm})
        wandb.config.update({'test_post_cluster_p_y_given_d_fro_norm': experiment.test_post_cluster_p_y_given_d_fro_norm})
        wandb.config.update({'test_post_cluster_acc_dd_uniform': experiment.test_post_cluster_acc_dd_uniform})
        wandb.config.update({'test_post_cluster_acc_dd_balanced': experiment.test_post_cluster_acc_dd_balanced})

        wandb.config.update({'test_post_cluster_p_y_given_d_l1_norm_uniform': experiment.test_post_cluster_p_y_given_d_l1_norm_uniform})
        wandb.config.update({'test_post_cluster_p_y_given_d_fro_norm_uniform': experiment.test_post_cluster_p_y_given_d_fro_norm_uniform})
        wandb.config.update({'test_post_cluster_p_y_given_d_l1_norm_balanced': experiment.test_post_cluster_p_y_given_d_l1_norm_balanced})
        wandb.config.update({'test_post_cluster_p_y_given_d_fro_norm_balanced': experiment.test_post_cluster_p_y_given_d_fro_norm_balanced})

        run.finish()