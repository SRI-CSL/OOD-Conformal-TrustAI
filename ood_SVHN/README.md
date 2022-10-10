# Get models 
wget https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0
wget https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0

# Get downloaded OOD datasets 
    mkdir data
    cd data

    wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
    tar -xf Imagenet_resize.tar.gz 
    wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
    tar -xf LSUN_resize.tar.gz 
    wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
    tar -xf iSUN.tar.gz 

    