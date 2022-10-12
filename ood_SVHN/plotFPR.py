import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('SVHN_ResNet_FPR.npy', 'rb') as f:
        resnet_fpr = np.load(f)
    with open('SVHN_DenseNet_FPR.npy', 'rb') as f:
        densenet_fpr = np.load(f)
    alpha = np.arange(0.01,1,0.01)
    plt.plot(alpha, resnet_fpr/100, '--.', color = 'blue')
    plt.plot(alpha, alpha, '--', color = 'red', label = '1 - TPR = alpha')
    plt.xlabel('alpha')
    plt.ylabel('False Alarm Probability (1 - TPR)')
    plt.legend()
    plt.savefig('SVHN_ResNet_FPR.png')
    plt.clf()
    plt.plot(alpha, densenet_fpr/100, '--.', color = 'blue')
    plt.plot(alpha, alpha, '--', color = 'red', label = '1 - TPR = alpha')
    plt.xlabel('alpha')
    plt.ylabel('False Alarm Probability (1 - TPR)')
    plt.legend()
    plt.savefig('SVHN_DenseNet_FPR.png')
    plt.clf()