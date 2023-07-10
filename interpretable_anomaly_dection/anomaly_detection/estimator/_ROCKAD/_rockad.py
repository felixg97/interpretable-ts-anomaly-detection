
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sktime.transformations.panel.rocket import Rocket

class NN:
    
    def __init__(self, 
            n_neighbors = 5, 
            n_jobs = 1,
            dist = 'euclidean',
            random_state=42, 
        ) -> None:
        
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.dist = dist
        self.random_state = random_state


    def fit(self, X):
        self.nn = NearestNeighbors(
            n_neighbors = self.n_neighbors,
            n_jobs = self.n_jobs,
            dist = self.dist,
            algorithm = 'ball_tree',
            )
        
        self.nn.fit(X)


    def predict_proba(self, X, y=None):
        scores = self.nn.kneighbors(X)
        scores = scores[0].mean(axis=1).reshape(-1,1)
        
        return scores
    


class ROCKAD():
    
    def __init__(self,
            n_estimators=10,
            n_kernels = 100,
            n_neighbors = None,
            n_jobs = 1,
            transform = True,
            random_state = 42,
        ) -> None:
        self.random_state = random_state
        self.transform = transform
        
        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.n_inf_cols = []
        
        self.estimator = NN
        self.transformer = Rocket(num_kernels = self.n_kernels, n_jobs = self.n_jobs, random_state = self.random_state)
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(standardize = False)
        
    
    
    def init(self, X):
        # rocket fitting and transforming
        Xt = self.rocket.fit_transform(X)
        
        if self.transform == True:
            # power transformation
            self.Xt_scaled = pd.DataFrame(self.power_transformer(Xt))
            
        else:
            self.Xt_scaled = pd.DataFrame(Xt)
        


    def fit_estimators(self, X):
        pass


    def fit(self, X):
        pass
    
    
    def predict_proba(self, X):
        pass
    
    
    def _check_inf_values(self, X):
        if np.isinf(X[X.columns[~X.columns.isin(self.n_inf_cols)]]).any(axis=0).any() : 
            self.n_inf_cols.extend(X.columns.to_series()[np.isinf(X).any()])
            self.fit_estimators()
            return True
    
    