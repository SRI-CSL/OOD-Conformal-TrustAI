### Install requirements in requirements.txt
### Download ood datasets to ./data from https://github.com/facebookresearch/odin
### Download model files for ResNet34 and Densenet from https://github.com/pokaxpoka/deep_Mahalanobis_detector
### To get results for ResNet34 run check_OOD_ResNet.py. Example command
    python check_OOD_ResNet.py --cuda --gpu 1 --ood_dataset SVHN
### To get results for DenseNet run check_OOD_DenseNet.py. Example command
    python check_OOD_DenseNet.py --cuda --gpu 1 --ood_dataset SVHN