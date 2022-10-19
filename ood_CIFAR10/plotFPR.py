import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('CIFAR10_ResNet_FPR.npy', 'rb') as f:
        resnet_fpr = np.load(f)
    with open('CIFAR10_DenseNet_FPR.npy', 'rb') as f:
        densenet_fpr = np.load(f)
    alpha = np.arange(0.01,0.2,0.01)
    resnet_fpr = resnet_fpr[:len(alpha)]
    densenet_fpr = densenet_fpr[:len(alpha)]
    plt.plot(alpha, resnet_fpr/100, '--o', color = 'blue')
    plt.plot(alpha, alpha, '--', color = 'red', label = 'Upper Bound')
    plt.xlabel('alpha', fontsize = 14)
    plt.ylabel('False Alarm Probability (1 - TPR)', fontsize = 14)
    plt.legend(prop={"size":14})
    plt.savefig('CIFAR10_ResNet_FPR.png')
    plt.clf()
    plt.plot(alpha, densenet_fpr/100, '--o', color = 'blue')
    plt.plot(alpha, alpha, '--', color = 'red', label = 'Upper Bound')
    plt.xlabel('alpha', fontsize = 14)
    plt.ylabel('False Alarm Probability (1 - TPR)', fontsize = 14)
    plt.legend(prop={"size":14})
    plt.savefig('CIFAR10_DenseNet_FPR.png')
    plt.clf()