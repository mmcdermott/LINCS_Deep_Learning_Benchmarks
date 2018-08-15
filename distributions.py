import numpy as np
from scipy.stats import uniform, beta, geom, dlaplace, rv_discrete


def to_numeric(x): return x[0] if type(x) is np.ndarray else x
def make_rvs(d): return {k: CategoricalRV(v) if type(v) is list else v for k, v in d.items()}
def sample_dict(dict_of_rvs): return {k: to_numeric(v.rvs(1)) for k, v in make_rvs(dict_of_rvs).items()}

class MixtureDistribution:
    def __init__(self, candidate_distributions, weights=None):
        self.candidate_distributions = candidate_distributions
        self.num_components = len(self.candidate_distributions)
        self.ws = weights if weights is not None else [1./self.num_components] * self.num_components
        self.distribution_selection = rv_discrete(
            name='mixture_components',
            values=(range(self.num_components), self.ws)
        )

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)

        dists = list(self.distribution_selection.rvs(size=b, random_state=random_state))
        counts = [0] * self.num_components
        #print('Num components: ', self.num_components)
        #print('weights: ', self.ws)
        for dist in dists:
            #print(dist)
            counts[dist] += 1

        vals = [None] * self.num_components
        for i, dist, count in zip(range(self.num_components), self.candidate_distributions, counts):
            if count > 0: 
                v = dist.rvs(count, random_state=random_state)
                #print(v)
                vals[i] = [v] if type(v) in [int, np.int32, np.int64, str] else list(v)

        samples = []
        for dist in dists:
            samples += [vals[dist].pop()]
        return samples if b > 1 else samples[0]
        #return samples (for layer dist)

class LayerDistribution:
    def __init__(self, num_layer_distribution, layer_size_distribution_fn):
        self.num_layer_distribution = num_layer_distribution
        self.layer_size_distribution_fn = layer_size_distribution_fn

    def __rv(self, random_state):
        n = self.num_layer_distribution.rvs(1, random_state=random_state)[0]
        assert n >= 0, "Invalid number of layers sampled!"

        if n == 0: return []

        rvs = [to_numeric(self.layer_size_distribution_fn(None, 0).rvs(1, random_state=random_state))]
        for layer in range(1, n):
            new_layer_size = to_numeric(
                self.layer_size_distribution_fn(rvs[-1], layer).rvs(1, random_state=random_state)
            )
            if new_layer_size < 2: new_layer_size = 2
            rvs.append(new_layer_size)

        return rvs

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)
        num_samples = b if type(b) is int else b[0]
        samples = [self.__rv(random_state) for _ in range(num_samples)]
        return tuple(samples)

class DeltaDistribution:
    def __init__(self, loc=0):
        self.x = loc

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)
        num_samples = b if type(b) in [int, np.int32, np.int64, str] else b[0]
        return [self.x] * num_samples

class CategoricalRV(MixtureDistribution):
    def __init__(self, options, weights=None):
        super(CategoricalRV, self).__init__([DeltaDistribution(x) for x in options], weights=weights)
