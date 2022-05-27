## Installation

- Install a recent version of Python 3 (we used Python 3.10.4).
- `pip install -r requirements.txt`
- You may receive errors for Pillow installation, if so follow instructions here: https://pillow.readthedocs.io/en/latest/installation.html
- Install ImageNet by the instructions at https://www.image-net.org/download.php and replace 'root folder' in ImageNet and ImageNetSubset classes in dataset.py with the root folder of the installation (one level above the train/validation split folders). The test dataset we use is composed of the validation dataset from ImageNet, the validation dataset is split out of the train dataset of ImageNet.
- Details on downloading the FieldGuide dataset can be found here https://sites.google.com/view/fgvc6/competitions/butterflies-moths-2019. Extract images from training.rar into 'FieldGuideAllImagesDownload'. Then run create_FieldGuide_directories.ipynb to create the FieldGuide-28 train, val and test directories. For FieldGuide-2 copy only classes 0 and 1 of Fieldguide-28 into a separate ~/fieldguide2/ directory, with the same train/val/test top level and the corresponding data folders from each split of classes 0-1 below each.
- Starting on line 274 of experiment_runner.py, replace "project" and "entity" with the appropriate project and entity for WandB.
- From https://github.com/wvangansbeke/Unsupervised-Classification, download CIFAR-10 SCAN Loss, CIFAR-100 SCAN Loss, and Imagenet-50 SCAN Loss pth.tar files into the repo top level directory.

## Operation

- `python experiment_runner.py --dataset [cifar10, cifar20, imagenet, fg2, fg28] --GPU [desired GPU, or leave blank if only CPU is available] --random_seed_wave [1,2,3,4]`
- Results will be available in WandB.
- Test acc in paper is reported as 'test_post_cluster_acc_dd_balanced'.
- $Q_{Y|D}$ err is computed by extracting 'test_post_cluster_p_y_given_d_l1_norm_balanced' and dividing it by the total number of entries in the $Q_{Y|D}$ matrix, which is (# domains) * (# true classes).


## Attributions:

Attribution is available in LICENSES_ATTRIBUTION.