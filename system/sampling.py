import numpy as np
import random

class FeedbackSampler:
    def __init__(self, n, probs=None):
        self.name = "feedback"
        self.n = n
        self.p = probs if probs is not None else np.ones(n)/float(n)
        self.explore = [i for i in range(n)]
        random.shuffle(self.explore)
        self.explored = False

    def sample(self, k, startup=0):
        if startup:
            k = self.n
        if len(self.explore) > 0 and not self.explored:
            sampled = self.explore[0:k]
            self.explore = list(set(self.explore) - set(sampled))
            self.last_sampled = np.sort(sampled), self.p[np.sort(sampled)]
            if len(self.explore)==0:
                self.explored = True
            return np.sort(np.array(sampled))
        else:
            nonzero_entries = sum(self.p > 0)
            if nonzero_entries >= k:
                sampled = np.random.choice(self.n, k, p=self.p, replace=False)
            else:
                sampled = np.random.choice(self.n, nonzero_entries, p=self.p, replace=False)
                remains = np.setdiff1d(np.arange(self.n), sampled)
                uniform_sampled = np.random.choice(remains, k-nonzero_entries, replace=False)
                sampled = np.concatenate((sampled, uniform_sampled))
            self.last_sampled =  np.sort(sampled), self.p[np.sort(sampled)]
            return np.sort(sampled)
        
    def update(self, probs, beta=1):
        self.p = (1-beta)*self.p + beta*probs

class UniformSampler:
    def __init__(self, n, probs=None):
        self.name = "uniform"
        self.n = n
        self.p = probs if probs is not None else np.ones(n)/float(n)

    def sample(self, k):
        if k == self.n:
            self.last_sampled = np.arange(self.n), self.p
            return np.arange(self.n)
        else:
            sampled = np.random.choice(self.n, k, p=self.p, replace=False)
            self.last_sampled = np.sort(sampled), self.p[np.sort(sampled)]
            return np.sort(sampled)
        
    def update(self, probs, beta=1):
        self.p = (1-beta)*self.p + beta*probs
        # print(self.p)