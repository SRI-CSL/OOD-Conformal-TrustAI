from tkinter import N
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Ks = [10]
    alphas = [0.05, 0.1]
    for K in Ks:
        for alpha in alphas:
            deltas = np.arange(0.01, 0.5, 0.005)
            n_cals = np.zeros(len(deltas))
            ctr = 0
            for delta in deltas:
                print(delta)
                n_search = np.arange(1000, 28000, 5)
                for n in n_search:
                    
                    sum_K = 0
                    for i in range(K):
                        sum_K += 1/(1+i)
                    alpha_prime = alpha / (2 * sum_K)
                
                    a = int((n+1)*alpha_prime/K)
                    b = n+1 - a
                    rv = beta(a,b)
                    mu = a/(a+b)

                    if rv.cdf(2*mu) >= 1 - delta/K**2:
                        print(n)
                        n_cals[ctr] = n 
                        ctr+=1
                        break

            plt.plot(deltas, n_cals)
            plt.xlabel('delta')
            plt.ylabel('n_cal')
            plt.savefig('n_cals_alpha_{}_K_{}.png'.format(alpha, K))  
            plt.clf()      
            

