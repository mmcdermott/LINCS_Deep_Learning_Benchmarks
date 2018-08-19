# LINCS (GSE92742) Deep Learning Benchmarks
A collection of benchmark tasks and results for the LINCS dataset. See our [BioKDD '18](http://home.biokdd.org/biokdd18/index.html) [Paper](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/camera-ready.pdf) or [Poster](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/biokdd-18-poster.pdf) for more information.

To access the fully processed dataset or graphs (though fully derived from public data), including all cross validation folds, exactly as was used in this work, and/or our hyperparameter search results, contact [mmd@mit.edu](mailto:mmd@mit.edu).

[`Hyperparameter Optimization Results.ipynb`](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/Hyperparameter%20Optimization%20Results.ipynb) contains a record of all hyperparameter search results and optimal findings--if you want to run it, first you need to download the data file (contact [mmd@mit.edu](mailto:mmd@mit.edu)) and place it in the right location, then it should be completely reproducible.

[`distributions.py`](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/distributions.py) contain helper functions to build the distributions used for the random hyperparameter search.

[`gcn_distributions.py`](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/gcn_distributions.py) contains code used to build the random samples of the GCNN parameters.

[`sklearn_classifiers.py`](https://github.com/mmcdermott/LINCS_Deep_Learning_Benchmarks/blob/master/sklearn_classifiers.py) contains code used to build the random samples for sklearn classifiers.

The code used to run the GCNNs is available [here](https://github.com/mmcdermott/cnn_graph), but the original source is

**Citation:** Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), Neural Information Processing Systems (NIPS), 2016.

**Github:** https://github.com/mdeff/cnn_graph
