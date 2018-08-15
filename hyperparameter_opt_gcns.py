RANDOM_STATE = 2
import random, numpy as np, tensorflow as tf
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)

from cnn_graph import models, graph, coarsening, utils
from plot_functions import *
from distributions import *
from graph_functions import *

import copy, math, argparse, os, shutil, networkx, pickle, scipy
from datetime import datetime
import pandas as pd
import gene_expression_modelling.parse_cmap as pcmp
from gene_expression_modelling.constants import *
from gene_expression_modelling.utils import *
import ml_toolkit.pandas_constructions as pdc
from sklearn.preprocessing import LabelEncoder

from scipy.stats import uniform, beta, geom, dlaplace, randint, poisson

BASE_PARAMS = {
    'dir_name': 'gcnn_final_classifier_{target}_{graph}_{dataset}_{run}_%s' % args.gpu,
    'num_epochs': args.epochs,
    'eval_frequency': 100000,

    # Building blocks.
    'filter': 'chebyshev5',
    'brelu': 'b1relu', #'b2relu' breaks variable sharing with multiple components.
}

PRIOR_BASE_PARAMS = copy.deepcopy(BASE_PARAMS)
PRIOR_BASE_PARAMS.update({
    'batch_size': 20,
    # Building blocks.
    'filter': 'chebyshev5',
    'brelu': 'b1relu',
    'pool': 'apool1',
    # Optimization.
    'regularization': 1e-2,
    'dropout': 0.5,
    'learning_rate': 6e-3,
    'decay_rate': 0.98,
    'momentum': 0.9,
    'decay_steps': 400,
})

PRIOR_FOUND_PARAMS = [{
    # Architecture.
    'Fs': [25],  # Number of graph convolutional filters.
    'Ks': [7],  # Polynomial orders.
    'ps': [2],    # Pooling sizes.
    'M': [100],  # Output dimensionality of fully connected layers.

    # Optimization.
    'regularization': 5e-2,
    'dropout': 0.5,
    'learning_rate': 1e-3,
    'decay_rate': 0.96,
    'momentum': 0.9,
}, {
    # Architecture.
    'Fs': [20],  # Number of graph convolutional filters.
    'Ks': [10],  # Polynomial orders.
    'ps': [2],    # Pooling sizes.
    'M': [100],  # Output dimensionality of fully connected layers.

    # Optimization.
    'regularization': 5e-2,
    'dropout': 0.5,
    'learning_rate': 5e-4,
    'decay_rate': 0.98,
    'momentum': 0.9,
}]

K = 15

GRAPHS = {'regnetworkdb': depickle(REG_GRAPH_FILE)}
HUMANBASE_GRAPH_DIR = os.path.join(BASE_DATA_DIR, 'Humanbase Graphs')
for graph_filename in os.listdir(HUMANBASE_GRAPH_DIR):
    graph_name = graph_filename[:-4] + '_humanbase'
    if not graph_filename.endswith('.pkl') or not graph_name in args.graphs_to_use:
        print("Skipping humanbase graph file", graph_filename)
        continue
    G = depickle(os.path.join(HUMANBASE_GRAPH_DIR, graph_filename))
    GRAPHS[graph_name] = G

GRAPHS = {k: v for k, v in GRAPHS.items() if k in args.graphs_to_use}

def build_layer_fcr(base_rv, decay_rate):
    return lambda prev, layer_num: (
        base_rv if prev is None else MixtureDistribution([
            poisson(prev * decay_rate, loc=2),
            DeltaDistribution(loc=prev),
            poisson(prev / decay_rate, loc=2),
        ])
    )

NUM_GRAPH_LAYERS_RV = geom(0.5)
DENSE_LAYERS_RV = LayerDistribution(
    num_layer_distribution=geom(0.5, loc=-1),
    layer_size_distribution_fn=build_layer_fcr(dlaplace(1e-1, loc=150), 0.5),
)

def constant_int_fcr(choices):
    n = len(choices)
    return lambda _1, _2: CategoricalRV(choices)

def filter_K_fcr(base_rv, delta_rate):
    return lambda prev, _: (
        base_rv if prev is None else MixtureDistribution([
            poisson(prev * delta_rate, loc=2),
            DeltaDistribution(loc=prev),
            poisson(prev / delta_rate, loc=2),
        ])
    )

STATIC_PARAM_DISTS = {
    # Optimization.
    #'brelu': CategoricalRV(['b1relu', 'b2relu']),
    'batch_size': randint(15, 150),
    'regularization': beta(1, 99),
    'dropout': beta(9, 6),
    'learning_rate': beta(1, 250),
    'decay_rate': beta(98, 2),
    'decay_steps': poisson(390),
    'momentum': beta(23.25, 1.75),
    'M': DENSE_LAYERS_RV,
    'pool': CategoricalRV(['apool1', 'mpool1', 'mpool1']), # 2/3 mpool1
}

EDGE_WEIGHT_CUTOFF_DIST = uniform(-0.8, 0.8)

GCLAYER_PARAM_DISTS = {
    # Architecture.
    'Fs': filter_K_fcr(randint(5, 45), 1.5),  # Number of graph convolutional filters.
    'Ks': filter_K_fcr(randint(2, 35), 1.5),  # Polynomial orders.
    'ps': constant_int_fcr([2, 2, 2, 4, 4])
}

def find(name, L): return next((e for n, e in L if n == name))
def get_levels(params): return int(sum((np.log2(p) for p in params['ps'][0]))) + 1

#if os.path.isfile(OPTIONS_FILENAME):
#    with open(OPTIONS_FILENAME, mode='rb') as f: all_params = pickle.load(f)
#else: all_params = [] # (dataset_name, graph_name, params)
all_params = []

if len(all_params) < 50:
    for dataset_name, dataset in DATASETS.items():
        for target_name, target_fn in TARGETS.items():
            num_classes = max(target_fn(dataset)) + 1
            for graph_name, G_full in GRAPHS.items():
                for sample in range(BUDGET):
                    edge_weight_cutoff = EDGE_WEIGHT_CUTOFF_DIST.rvs(1)[0]
                    if graph_name != 'Correlation': edge_weight_cutoff = abs(edge_weight_cutoff)
                    if graph_name == 'regnetworkdb':
                        edge_weight_cutoff = 0.0
                        G = G_full
                    else:
                        print("Processing Graph {name} @ {e} for {dataset} -> {target}".format(
                            name=graph_name, e=edge_weight_cutoff, target=target_name, 
                            dataset=dataset_name
                        ))
                        G = cutoff_graph(G_full, edge_weight_cutoff)

                    _, nodelists, _ = get_components(G)
                    num_components = len(nodelists)
                    if num_components == 0:
                        print(
                            "Completely disconnected graph {name} found "
                            "for edge weight cutoff {e}!".format(name=graph_name, e=edge_weight_cutoff)
                        )
                        continue
                    smallest_component_size = min(len(l) for l in nodelists)

                    params = copy.deepcopy(BASE_PARAMS)
                    for k, rv in STATIC_PARAM_DISTS.items():
                        v = rv.rvs(1)
                        params[k] = v if type(v) in [
                            int, float, np.float32, np.float64, np.int64, np.int32, str
                        ] else v[0]

                    params['M'].append(num_classes)

                    num_graph_layers = int(NUM_GRAPH_LAYERS_RV.rvs(1)[0])
                    while num_graph_layers > np.log2(smallest_component_size) - 1:
                        print('num_graph_layers', num_graph_layers)
                        num_graph_layers = int(NUM_GRAPH_LAYERS_RV.rvs(1)[0])

                    for k, rv_fn in GCLAYER_PARAM_DISTS.items():
                        if k == 'ps' and num_graph_layers >= math.floor(np.log2(smallest_component_size)) - 1:
                            params[k] = [[2] * num_graph_layers] * num_components
                            continue
                        rvs = [int(rv_fn(None, 0).rvs(1))]
                        for layer in range(1, num_graph_layers):
                            v = rv_fn(rvs[-1], layer).rvs(1)
                            rvs.append(v if type(v) in [int, np.int64, np.int32, str] else int(v[0]))
                        params[k] = [rvs] * num_components

                    while get_levels(params) > np.log2(smallest_component_size):
                        print(get_levels(params), np.log2(smallest_component_size), 're-gen')
                        k, rv_fn = 'ps', GCLAYER_PARAM_DISTS['ps']
                        rvs = [int(rv_fn(None, 0).rvs(1))]
                        for layer in range(1, num_graph_layers):
                            v = rv_fn(rvs[-1], layer).rvs(1)
                            rvs.append(v if type(v) in [int, np.int64, np.int32, str] else int(v[0]))
                        params[k] = [rvs] * num_components
                    all_params.append(
                        (dataset_name, graph_name, edge_weight_cutoff, target_name, params.copy())
                    )
