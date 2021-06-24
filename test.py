#! /usr/bin/env python3

from dataclasses import dataclass
import numpy as np 
from matplotlib import pyplot as plt 
from statsmodels.api import OLS 

@dataclass 
class generate_random():
    nrep: int
    distr:  str

    def generate(self, **kwargs):
        if self.distr == 'Normal':
            m = kwargs['m']
            s = kwargs['s']
            nrep = self.nrep
            return np.random.normal(m, s, nrep)

@dataclass
class simulate_data():
    nrep: int
    err_type: str


    def linear(self, betas, x=None):
        if x is None: 
            print("X not provided. Assuming random N(0,1)")
            
            nfact = betas.shape[0]
            
            g = generate_random(self.nrep, "Normal")

            X = np.array([g.generate(**{'m': 0, 's': 1}) for _ in range(0, nfact)])
            
            eps = generate_random(self.nrep, self.err_type).generate(**{'m': 0, 's': 1})
            
            return (betas.reshape(1, nfact)@X + eps, X)

if __name__ == '__main__':
    sm = simulate_data(1000, 'Normal')

    betas = np.array([1, 1.5])

    print(betas)

    y, X = sm.linear(betas)

    print("shape of y is (%d, %d)" % y.shape)

    print(y)


    plt.plot(y)
    plt.show()
