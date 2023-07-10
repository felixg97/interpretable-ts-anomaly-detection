import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
import pandas as pd

class NN: 
    
    def __init__(
            self,
            random_state = 42,
            n_neighbors = 5,
            n_jobs = 1,
            dist = 'euclidean',
            decision_function = 'mean'
            ): 

        self.n_neighbors = n_neighbors 
        self.rs = random_state
        self.n_jobs = n_jobs
        self.dist = dist
        self.decision_function = decision_function
        
        
    def __str__(self): # create object name based on parameters
        
        def apply_leading_zero(number, threshold):
            if number < threshold: 
                placeholder = '0'
            else: 
                placeholder = ''
            return placeholder
        
        name = ('NN' + apply_leading_zero(self.n_neighbors, 10)
                + str(self.n_neighbors))
        
        return name
        
    def fit(self, X): 
        self.nn = NearestNeighbors(
            n_neighbors = self.n_neighbors,
            n_jobs = self.n_jobs,
            dist = self.dist,
            algorithm = 'ball_tree'
            )
        self.nn.fit(X)
        
    def predict_proba(self, X, y=None): 
        scores = self.nn.kneighbors(X)
        scores = scores[0].mean(axis=1).reshape(-1,1)
        return scores


class ROCKAD: 
    
    def __init__(
            self,
            n_estimators=10,
            n_kernels = 100,
            n_neighbors = None,
            random_state = 42,
            random_state_bootstrap = None,
            transform = True,
            n_jobs =1): 
        
        
        self.random_state = random_state
        self.transform = transform
        
        
        if random_state_bootstrap == None:
            self.random_state_bootstrap = self.random_state
        else:
            self.random_state_bootstrap = random_state_bootstrap
        
        self.estimator = NN
        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.n_neighbors = n_neighbors 
        self.n_jobs = n_jobs
        self.sscaler = StandardScaler()
        
        # self.rocket = Rocket( TODO: AWAY
        #     num_kernels = self.n_kernels,
        #     random_state = self.random_state,
        #     n_jobs = self.n_jobs
        #     )
        
        self.scaler_init = PowerTransformer(standardize = False)
        self.n_inf_cols = []
        
    def __str__(self): 

        
        def apply_leading_zero(number, threshold):
            if number < threshold: 
                placeholder = '0'
            else: 
                placeholder = ''
            return placeholder
        
        name = ('ROCKAD_'
            + apply_leading_zero(self.n_kernels, 10000)
            + apply_leading_zero(self.n_kernels, 1000)
            + apply_leading_zero(self.n_kernels, 100)
            + apply_leading_zero(self.n_kernels, 10)
            + str(self.n_kernels)
            + '_' + apply_leading_zero(self.n_estimators, 10)
            + str(self.n_estimators) + '_' 
            + apply_leading_zero(self.n_neighbors, 10)
            + str(self.n_neighbors))
        
        return name
    
    def copy(self): 
        return self
    
    def init_rocket_scaler(self, X):
        
        if self.transform == True: 
            
            # Xn = from_2d_array_to_nested(X)
            
            #rocket fitting and transformation
            # Xt_rocket = self.rocket.fit_transform(Xn)
            Xt_rocket = X # TODO: AWAY for CaseStudy
            
            # Scale and cache training data. Caching is required for refitting 
            # the estimators if a column becomes infinite after powertransf.
            self.Xt_scaled = pd.DataFrame(self.scaler_init.fit_transform(Xt_rocket))
            
        else: 
            self.Xt_scaled = X
            
        
    def fit_estimators(self): 
        
        if self.transform == True:
            # Check for infinite columns and get indices
            self.check_inf(self.Xt_scaled)
            
            self.Xt_scaled = self.Xt_scaled[self.Xt_scaled.columns[~self.Xt_scaled.columns.isin(self.n_inf_cols)]]
            
            # Fit Standardscaler
            Xt_scaled = self.sscaler.fit_transform(self.Xt_scaled) 
            
            Xt_scaled = pd.DataFrame(Xt_scaled, columns=self.Xt_scaled.columns)
            
            self.check_inf(Xt_scaled)
            
            Xt_scaled = Xt_scaled[Xt_scaled.columns[~Xt_scaled.columns.isin(self.n_inf_cols)]]
            
            
            Xt_scaled = Xt_scaled.astype(np.float32).to_numpy()

        else: 
            Xt_scaled = self.Xt_scaled.astype(np.float32)
            
        self.list_baggers = [] 
        for n_estimator in range(self.n_estimators): 
            
            #init estimator
            estimator = self.estimator(
                n_neighbors = self.n_neighbors,
                n_jobs = self.n_jobs
                )

            #bootstrap aggregation
            X_sample = resample(
                Xt_scaled,
                replace = True,
                n_samples=None,
                random_state = self.random_state_bootstrap + n_estimator,
                stratify = None
                )
            
            #fit estimator and append to estimator list
            estimator.fit(X_sample)
            self.list_baggers.append(estimator)
            
    def fit(self, X): 
        self.init_rocket_scaler(X)
        self.fit_estimators()


    def check_inf(self, X):
        
        if np.isinf(X[X.columns[~X.columns.isin(self.n_inf_cols)]]).any(axis=0).any() : 
            self.n_inf_cols.extend(X.columns.to_series()[np.isinf(X).any()])
            self.fit_estimators()
            return True
       

            
    def predict_proba(self, X, y=None): 
        y_scores = np.zeros((len(X), self.n_estimators))
        
        if self.transform == True: 
            
            # Xn = from_2d_array_to_nested(X) # TODO: AWAY for CaseStudy
            
            #rocket transform with fitted parameters
            # Xt_rocket = self.rocket.transform(Xn) # TODO: AWAY for CaseStudy
            Xt_rocket = X # TODO: for CaseStudy
            
            # power transform with fitted parameters and check for infinity
            Xt_scaled = self.scaler_init.transform(Xt_rocket)
            Xt_scaled = pd.DataFrame(Xt_scaled)
            
            self.check_inf(Xt_scaled)
            Xt_scaled = Xt_scaled[Xt_scaled.columns[~Xt_scaled.columns.isin(self.n_inf_cols)]]
            Xt_scaled_ = Xt_scaled
            
            Xt_scaled = self.sscaler.transform(Xt_scaled_)
            Xt_scaled = pd.DataFrame(Xt_scaled, columns=Xt_scaled_.columns)
            
            self.check_inf(Xt_scaled)
            Xt_scaled = Xt_scaled[Xt_scaled.columns[~Xt_scaled.columns.isin(self.n_inf_cols)]].astype(np.float32).to_numpy()
        else: 
            Xt_scaled = X.astype(np.float32)
            
        
        # loop through the fitted estimators and predict probability 
        #for every estimator
        # print(len(self.list_baggers))
        for i, bagger in enumerate(self.list_baggers): 
            scores = bagger.predict_proba(Xt_scaled).squeeze()
            y_scores[:,i] = scores
            

        return y_scores.mean(axis = 1)