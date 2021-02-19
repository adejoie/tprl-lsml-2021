# TODO: should I store the data in the GPU?
import numpy as np

class ReplayBuffer(object):
    """ Expects tuples of (state, next_state, action, reward, done) """

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def can_sample(self, batch_size):
        return batch_size <= len(self.storage)
    
    def sample(self, batch_size):
        keys = np.random.choice(len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[key] for key in keys]

        x, y, u, r, d = [], [], [], [], []
        for (X, Y, U, R, D) in batch: 
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

