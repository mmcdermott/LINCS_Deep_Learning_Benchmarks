from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import calibration_curve
from sklearn.gaussian_process.kernels import *
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from scipy.stats import beta, dirichlet, truncexpon, lognorm, geom, poisson, uniform, pearsonr, dlaplace,\
                        rv_discrete, randint

from distributions import *

def corr(x, y): return pearsonr(x, y)[0]

NB = 'Naive Bayes'
LIN = 'Linear Classifier'
KNN = 'K-neighbors'
DT = 'Decision Tree'
RF = 'Random Forest'
ADA = 'AdaBoost'
KSVC = 'Kenerl SVC'
FFANN = 'SKlearn Fully Connected ANN'
GPC = 'Gaussian Processes'
QDA = 'Quadratic Discriminant Analysis'
NKSVC = 'Nu Kenerl SVC'

CLASSIFIER_NAMES = {
    'Naive Bayes': GaussianNB,
    'Linear Classifier': SGDClassifier,
    'K-neighbors': KNeighborsClassifier,
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'AdaBoost': AdaBoostClassifier,
    'Kenerl SVC': SVC,
    'SKlearn Fully Connected ANN': MLPClassifier,
    'Gaussian Processes': GaussianProcessClassifier,
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis,
    'Nu Kenerl SVC': NuSVC,
}

CLASSIFIRES_PARAM_DISTS = {
    #'Naive Bayes': {}, No parameters, so no point tuning.
    'Linear Classifier': {
        'loss':          ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty':       ['l2', 'l1', 'elasticnet'],
        'alpha':         truncexpon(8),
        'l1_ratio':      uniform(0, 1),
        'learning_rate': ['optimal', 'invscaling', 'constant'],
        'eta0':          lognorm(3, scale=np.exp(-6)),
        'power_t':       lognorm(1.5, scale=np.exp(-2)),
    },
    'K-neighbors': {
       'n_neighbors': geom(0.1),
       'weights': ['uniform', 'distance'],
       'p': geom(0.3),
       'metric': ['minkowski', 'chebyshev', 'canberra', corr],
    },
    'Decision Tree': {
         'criterion': ['gini', 'gini', 'entropy'],
         'max_depth': [None, 2, 5, 10, 25, 100],
         'splitter': ['best', 'best', 'random'],
         'min_samples_split': geom(0.9, loc=1),
         'min_samples_leaf': geom(0.9),
         'min_weight_fraction_leaf': beta(1, 100, scale=0.5),
         'max_leaf_nodes': [None, 2, 5, 10, 25, 100],
         'min_impurity_decrease': beta(1, 1000),
         'max_features': [None, 'sqrt', 'log2', 10, 25, 100, 250],
    },
    'Random Forest': {
        'n_estimators': MixtureDistribution([poisson(50), poisson(200), poisson(400)]),
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 25, 100],
        'min_samples_split': geom(0.8, loc=1),
        'min_samples_leaf': geom(0.8),
        'min_weight_fraction_leaf': beta(1, 100, scale=0.5),
        'max_leaf_nodes': [None, 25, 100, 250, 500],
        'min_impurity_decrease': beta(1, 1000),
    },
    'AdaBoost': {
        'algorithm': ['SAMME'],
        'base_estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(),
            SGDClassifier(alpha=0.02, penalty='l2', loss='log', max_iter= 100, n_jobs= -1, tol=1e-5),
        ],
        'n_estimators': MixtureDistribution([poisson(50), poisson(200), poisson(400)]),
        'learning_rate': beta(1, 999),
    },
    'Kenerl SVC': {
        'C': beta(1, 10),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': geom(0.5, loc=1),
        'coef0': truncexpon(8),
        'decision_function_shape': ['ovo', 'ovr'],
    },
    'SKlearn Fully Connected ANN': {
        'alpha': truncexpon(8),
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': truncexpon(2),
        'power_t': uniform(0, 1),
        'momentum': beta(90, 10),
        'nesterovs_momentum': [True, False],
        'early_stopping': [True, False],
        'max_iter': randint(100, 400),
        'beta_1': beta(90, 10),
        'beta_2': beta(999, 1),
        'epsilon': beta(1, 100000000),
        'hidden_layer_sizes': MixtureDistribution([
            LayerDistribution(
                geom(0.5), lambda _, layer: poisson(50) if layer > 4 else poisson(978 - (200 * layer))
            ),
            LayerDistribution(
                geom(0.5), lambda size, _: poisson(978) if size is None else poisson(0.75 * size)
            ),
            LayerDistribution(
                geom(0.5), 
                lambda _1, _2: MixtureDistribution([
                    poisson(200), poisson(500), poisson(978)
                ]),
            ),
        ]),
    },
    'Gaussian Processes': {
      'kernel': [
          ConstantKernel(),
          DotProduct(),
          ExpSineSquared(),
          #Exponentiation(),
          Matern(),
          RBF(),
          RationalQuadratic(),
          Sum(WhiteKernel(), RationalQuadratic()),
          Sum(WhiteKernel(), RBF()),
          Sum(WhiteKernel(), ExpSineSquared()),
          Sum(WhiteKernel(), Matern()),
          Sum(WhiteKernel(), DotProduct()),
      ],
      'n_restarts_optimizer': geom(0.9),
    },
    'Quadratic Discriminant Analysis': {
        'reg_param': beta(1, 9),
    },
    'Nu Kenerl SVC': {
        'nu': beta(3, 3),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': geom(0.5, loc=1),
        'coef0': truncexpon(8),
        'decision_function_shape': ['ovo', 'ovr'],
    },
}

PRIOR_BEST_PARAMS = {
    'AdaBoost': [
        {
            'algorithm': 'SAMME',
            'base_estimator': DecisionTreeClassifier(
                class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'
            ),
            'learning_rate': 0.0032276375871971253,
            'n_estimators': 51,
            'random_state': 101
        }, {
            'algorithm': 'SAMME',
            'base_estimator': DecisionTreeClassifier(
                class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'
            ),
            'learning_rate': 0.0032276375871971253,
            'n_estimators': 51,
            'random_state': 101
        }
    ],
    'SKlearn Fully Connected ANN': [
        {
            'activation': 'relu',
            'alpha': 1.161588809744823,
            'beta_1': 0.83987092590021484,
            'beta_2': 0.99758337703103828,
            'early_stopping': False,
            'epsilon': 3.3700138825192237e-08,
            'hidden_layer_sizes': [976],
            'learning_rate': 'invscaling',
            'learning_rate_init': 0.40472851021894218,
            'max_iter': 1000,
            'momentum': 0.90655947674591775,
            'nesterovs_momentum': True,
            'power_t': 0.32066113809042096,
            'random_state': 101,
            'solver': 'lbfgs',
            'tol': 1e-05
        },
        {'max_iter': 350, 'activation': 'relu', 'learning_rate_init': 0.97468480931754664, 'momentum': 0.8754652850569925,
            'power_t': 0.35185231877758516, 'solver': 'lbfgs', 'nesterovs_momentum': False, 'alpha':
            0.33788243622220415, 'epsilon': 9.9821674554640334e-09, 'beta_1': 0.84281673142196667, 'beta_2':
            0.99996279847461822, 'learning_rate': 'constant', 'hidden_layer_sizes': 997, 'early_stopping':
            True},
        {'max_iter': 350, 'epsilon': 4.7124355839071608e-08, 'momentum': 0.87578479673634479, 'beta_2': 0.99754630341109385,
            'beta_1': 0.92942558433548428, 'activation': 'relu', 'learning_rate_init': 0.98606268473920389,
            'hidden_layer_sizes': [946, 193], 'power_t': 0.88708954128000406, 'learning_rate': 'constant',
            'solver': 'lbfgs', 'alpha': 1.1052923539020645, 'early_stopping': False, 'nesterovs_momentum':
            True},
        {'max_iter': 350, 'activation': 'relu', 'momentum': 0.9065594767459177, 'early_stopping': False, 'hidden_layer_sizes':
            [976], 'nesterovs_momentum': True, 'learning_rate_init': 0.4047285102189422, 'beta_1':
            0.8398709259002148, 'beta_2': 0.9975833770310383, 'learning_rate': 'invscaling', 'tol': 1e-05,
            'alpha': 1.161588809744823, 'power_t': 0.32066113809042096, 'random_state': 101, 'solver':
            'lbfgs', 'epsilon': 3.370013882519224e-08, 'max_iter': 1000},
        {'max_iter': 350, 'beta_1': 0.86439458708505201, 'learning_rate_init': 0.055303746246958198, 'hidden_layer_sizes':
            [997], 'solver': 'sgd', 'momentum': 0.86709953459951516, 'nesterovs_momentum': True, 'power_t':
            0.22601411862522569, 'beta_2': 0.99855467373634565, 'learning_rate': 'invscaling', 'alpha':
            0.82006406848636726, 'epsilon': 8.5796638788282495e-09, 'early_stopping': False, 'activation':
            'relu'},
        {'max_iter': 350, 'beta_1': 0.93117401252301835, 'activation': 'tanh', 'epsilon': 1.1181028148969185e-08, 'beta_2':
            0.99920214088237735, 'solver': 'sgd', 'early_stopping': True, 'nesterovs_momentum': False,
            'momentum': 0.93188725719500154, 'alpha': 0.98570035796112321, 'learning_rate_init':
            0.20724074137333875, 'power_t': 0.037268324764610727, 'learning_rate': 'adaptive',
            'hidden_layer_sizes': [896, 679]},
        {'max_iter': 350, 'learning_rate': 'adaptive', 'beta_2': 0.99927254080997729, 'activation': 'logistic', 'beta_1':
            0.93453259091631535, 'hidden_layer_sizes': [975], 'solver': 'sgd', 'power_t': 0.74385471896258193,
            'early_stopping': True, 'epsilon': 1.206153958542538e-08, 'momentum': 0.90981549087918234,
            'learning_rate_init': 0.37686906743651638, 'nesterovs_momentum': True, 'alpha':
            0.19856153391293785},
    ],
    'Decision Tree': [
        {
            'criterion': 'gini',
            'max_depth': 100,
            'max_features': None,
            'max_leaf_nodes': 25,
            'min_impurity_decrease': 0.00062929163011517947,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0010639961338479592,
            'random_state': 101,
            'splitter': 'best'
        }, {
            'criterion': 'entropy',
            'max_depth': 10,
            'max_features': None,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0012257673197162205,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0020780796100258966,
            'random_state': 101,
            'splitter': 'best'
        }
    ],
    'K-neighbors': [
        {
            'metric': 'minkowski',
            'n_jobs': -1,
            'n_neighbors': 22,
            'p': 1,
            'weights': 'uniform'
        }, {
            'metric': 'canberra',
            'n_jobs': -1,
            'n_neighbors': 1,
            'p': 3,
            'weights': 'distance'
        }
    ],
    'Kenerl SVC': [
        {
            'C': 0.20344917816268288,
            'coef0': 0.7804057829638239,
            'decision_function_shape': 'ovr',
            'degree': 4,
            'kernel': 'poly',
            'max_iter': 1000,
            'random_state': 101,
            'tol': 1e-05
        }, {
            'C': 0.20344917816268288,
            'coef0': 0.7804057829638239,
            'decision_function_shape': 'ovr',
            'degree': 4,
            'kernel': 'poly',
            'max_iter': 1000,
            'random_state': 101,
            'tol': 1e-05
        }
    ],
    'Linear Classifier': [
        {
            'alpha': 0.0012311722512335377,
            'eta0': 0.00031685190815167104,
            'l1_ratio': 0.40558624196055393,
            'learning_rate': 'invscaling',
            'loss': 'log',
            'max_iter': 1000,
            'n_jobs': -1,
            'penalty': 'l1',
            'power_t': 0.18401717078886631,
            'random_state': 101,
            'tol': 1e-05
        }, {
            'alpha': 1.4382845341830468,
            'eta0': 0.00028857482328110371,
            'l1_ratio': 0.97207196286746378,
            'learning_rate': 'invscaling',
            'loss': 'squared_hinge',
            'max_iter': 1000,
            'n_jobs': -1,
            'penalty': 'l2',
            'power_t': 0.17584874098105796,
            'random_state': 101,
            'tol': 1e-05
        }, {
            'alpha': 0.02,
            'loss': 'huber',
            'max_iter': 1000,
            'n_jobs': -1,
            'penalty': 'l2',
            'tol': 1e-05
        }, {
            'alpha': 0.02,
            'loss': 'huber',
            'max_iter': 1000,
            'n_jobs': -1,
            'penalty': 'l2',
            'tol': 1e-05
        }
    ],
    'Random Forest': [
        {
            'criterion': 'entropy',
            'max_depth': 100,
            'max_leaf_nodes': 250,
            'min_impurity_decrease': 0.00029936337875902862,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0064574839705492668,
            'n_estimators': 389,
            'n_jobs': -1,
            'random_state': 101
        }, {
            'criterion': 'entropy',
            'max_depth': 10,
            'max_leaf_nodes': 500,
            'min_impurity_decrease': 0.0015687848935000061,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0010384615399753779,
            'n_estimators': 405,
            'n_jobs': -1,
            'random_state': 101
        }
    ],
    'Naive Bayes': [{}],
}
