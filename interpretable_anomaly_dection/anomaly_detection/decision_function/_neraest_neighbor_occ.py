
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import distance_metrics


class NearestNeighbor():
    
    def __init__(self, dist="euclidean"):
        self.scores_train = None
        self.dist = None
        
        if dist == "euclidean":
            self.dist = euclidean_distances
        elif dist in distance_metrics().keys():
            self.dist = dist[dist]
        elif dist in distance_metrics().values():
            self.dist = dist
        elif False:
            # TODO: allow time series distance functions
            pass
        else:
            raise Exception("Distance metric not supported.")
    
    
    def fit(self, scores_train):
        _scores_train = scores_train
        
        if type(_scores_train) is not np.array:
            _scores_train = np.array(scores_train.copy())
        
        if len(_scores_train.shape) == 1:
            _scores_train = _scores_train.reshape(-1, 1)
        
        self.scores_train = _scores_train
    
        
    def predict(self, scores_test):
        """
        Here : -1 indicates anormal, 1 indicates normal. 
        Per definition: 0 indicates anormal, 1 indicates normal.
        """
        
        predictions = []
        for score in scores_test:
            predictions.append(self.predict_score(score))
        return np.array(predictions)
    
    
    def predict_score(self, anomaly_score):
        prediction = None
        
        anomaly_score_arr = np.array([anomaly_score for i in range(len(self.scores_train))])
        
        _scores_train = self.scores_train.copy().reshape(-1, 1)
        anomaly_score_arr = anomaly_score_arr.reshape(-1, 1)
        nearest_neighbor_idx = np.argmin(self.dist(anomaly_score_arr, _scores_train))
        
        _scores_train = np.delete(_scores_train, nearest_neighbor_idx).reshape(-1, 1)
        
        nearest_neighbor_score = self.scores_train[nearest_neighbor_idx]
        neares_neighbot_score_arr = np.array([nearest_neighbor_score for i in range(len(_scores_train))])
        nearest_neighbor_score_arr = neares_neighbot_score_arr.reshape(-1, 1)
        
        nearest_nearest_neighbor_idx = np.argmin(self.dist(nearest_neighbor_score_arr, _scores_train))
        nearest_nearest_neighbor_score = _scores_train[nearest_nearest_neighbor_idx][0]
        
        prediction = self.indicator_function(
            anomaly_score, nearest_neighbor_score, nearest_nearest_neighbor_score)
        
        return prediction
    
    
    def indicator_function(self, z_score, nearest_score, nearest_of_nearest_score):
        
        # make it an array and reshape it to calculate the distance
        z_score_arr = np.array(z_score).reshape(1, -1)
        nearest_score_arr = np.array(nearest_score).reshape(1, -1)
        nearest_of_nearest_score_arr = np.array(nearest_of_nearest_score).reshape(1, -1)
        
        numerator = self.dist(z_score_arr, nearest_score_arr)
        denominator = self.dist(nearest_score_arr, nearest_of_nearest_score_arr)
        
        # error handling for extreme cases
        if numerator == 0:
            return 1
        elif denominator == 0:
            return -1
        else:
            return 1 if (numerator/denominator) <= 1 else -1