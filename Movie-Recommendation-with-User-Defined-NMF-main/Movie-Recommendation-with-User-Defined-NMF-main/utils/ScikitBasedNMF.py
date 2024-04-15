import numpy as np
from sklearn.decomposition import NMF

class UserDefinedNMF:

    def __init__(self, n_components=15, init='random', solver='cd', 
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio

        self.model = NMF(n_components=self.n_components, init=self.init, solver=self.solver, 
                         beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
                         random_state=self.random_state, alpha_W=self.alpha_W, 
                         alpha_H=self.alpha_H, l1_ratio=self.l1_ratio)
        
    def fit(self, X):
        self.model.fit(X)
        return self
    
    def transform(self, X):
        return self.model.transform(X)
    
    def fit_transform(self, X):
        return self.model.fit_transform(X)
    
    def reconstruct(self, X):

        W = self.model.transform(X)
        H = self.model.components_

        return np.dot(W, H)