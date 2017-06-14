# Python2
"""
Created on Mon May 29 20:46:04 2017

@author: GAURAV
"""

''' Reinforcement Learning with Python - Udemy 
    Section 2 Lecture 8.
    
    comparing_epsilons.py
'''

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,m):
        self.m = m
        self.mean = 10  # Optimistic Initial Values. Allows faster convergence
        self.N = 0
        
    def pull(self):
        return np.random.randn() +1 + self.m
        
    def update(self,x):
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean + 1.0/self.N*x
        
def run_experiment(m1,m2,m3,eps,N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    j = 0
    random_choic = 0
    data = np.empty(N)
    for i in xrange(N):
        p = np.random.random()
        if p < eps:
            random_choic += 1
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x
    
    cumulative_avg = np.cumsum(data)/(np.arange(N)+1)

    plt.plot(cumulative_avg)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    print [b.mean for b in bandits]
    
    return bandits,cumulative_avg
    
if __name__ == "__main__":
    band1,c_1 = run_experiment(1.0,2.0,3.0,0.1,100000)
    band2,c_05 = run_experiment(1.0,2.0,3.0,0.05,100000)
    band3,c_01 = run_experiment(1.0,2.0,3.0,0.01,100000)
    
    plt.figure()
    plt.plot(c_1,label='eps=0.1')
    plt.plot(c_05,label='eps=0.05')
    plt.plot(c_01,label='eps=0.01')  
    plt.legend()
    plt.show()
        
