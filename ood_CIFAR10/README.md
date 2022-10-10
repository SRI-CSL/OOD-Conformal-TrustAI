### Install requirements in requirements.txt
### Download ood datasets to ./data from https://github.com/facebookresearch/odin
    mkdir data
    cd data

    wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
    tar -xf Imagenet_resize.tar.gz 
    wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
    tar -xf LSUN_resize.tar.gz 
    wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
    tar -xf iSUN.tar.gz 
### Download model files for ResNet34 and Densenet from https://github.com/pokaxpoka/deep_Mahalanobis_detector
    wget https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth
    wget https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth
    
### To get results for ResNet34 run check_OOD_ResNet.py. Example command
    python check_OOD_ResNet.py --cuda --gpu 0 --net ./resnet_cifar10.pth --n 5 --ood_dataset LSUN --proper_train_size 45000 --trials 1
### To get results for DenseNet run check_OOD_DenseNet.py. Example command
    python check_OOD_DenseNet.py --cuda --gpu 1 --net ./densenet_cifar10.pth --n 5 --ood_dataset LSUN --proper_train_size 45000 --trials 1
