# Python2
"""
Created on Thu Jun 08 21:07:36 2017

@author: GAURAV
"""

''' Reinforcement Learning with Python - Udemy 
    Section 2 Lecture 10.
    
    exploration-eploitation using UCB1
    ucb1.py
'''
import numpy as np
import matplotlib.pyplot as plt

Bandits_var = {k: [] for k in range(3)}

class Bandit:
    def __init__(self,m):
        self.m = m
        self.mean = 0
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m
        
    def update(self,x):
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean + 1.0/self.N*x
        
def ucb(mean,N,Nj):
    return mean + np.sqrt(2*np.log(N)/(Nj+1e-2))
        
def run_experiment(m1,m2,m3,N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    j = 0
    data = np.empty(N)
    for i in xrange(N):
        j = np.argmax([ucb(b.mean,i,b.N) for b in bandits])
        for k in range(3):
            Bandits_var[k].append(np.sqrt(2*np.log(i)/(bandits[k].N+1e-2)))
        
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
    run_experiment(1,2,3,10000)