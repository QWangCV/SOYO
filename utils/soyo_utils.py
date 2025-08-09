import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn

def feature_compression(features, n_components):
    gmm = GaussianMixture(n_components=n_components, 
                          covariance_type='full', 
                          init_params='kmeans', 
                          verbose=1)
    gmm.fit(features)

    return (gmm.means_, gmm.covariances_, gmm.weights_) # [k, dim], [k, dim, dim], [k]


def random_sampling(n_samples, compression, n_components):
    (gmm_means, gmm_covariances, gmm_weights) = compression
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.means_ = gmm_means
    gmm.covariances_ = gmm_covariances
    gmm.weights_ = gmm_weights
    reconstructed_features = gmm.sample(n_samples)[0]

    return reconstructed_features


class soyo_network(nn.Module):
    def __init__(self, args):
        super(soyo_network, self).__init__()
        self.soyo_clf = nn.Linear(args['soyo_dim'], args['total_sessions'])

    def forward(self, image_features):
        domain_logits = self.soyo_clf(image_features)
        
        return domain_logits
