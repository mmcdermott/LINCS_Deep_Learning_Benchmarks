GSE92742 = 'GSE92742'
GSE92742_PROSTATE = 'GSE92742 (prostate only)'

PERT = 'Perturbagen'
MOA = 'MOA'
SITE = 'Primary Site'
SUBTYPE = 'Subtype'

MOA_PROSTATE = 'Prostate MOA'


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
GCNN = 'Graph Convolutional Neural Network'

FINAL_BEST_PARAMS = {
    GSE92742: {
        MOA: {
            GCNN: [
                {'dropout': 0.6980409669798152, 'regularization': 0.01087902135723254, 'Fs': [[9]], 'ps':
                    [[2]], 'momentum': 0.8789501097227141, 'M': [137, 49], 'learning_rate':
                    0.0012265138868632257, 'decay_rate': 0.9914003298632604, 'num_epochs': 350, 'decay_steps':
                    405, 'pool': 'apool1', 'batch_size': 92, 'Ks': [[7]]},
            ],
            FFANN: [
                {'batch_size': 92, 'beta_1': 0.919473939313036, 'activation': 'relu', 'alpha': 1.6879174708893805, 'epsilon':
                    9.7045902021921481e-10, 'solver': 'sgd', 'power_t': 0.33021989249044037,
                    'learning_rate_init': 0.10898026569061127, 'hidden_layer_sizes': [955], 'max_iter': 164,
                    'beta_2': 0.99919431656457547, 'learning_rate': 'adaptive', 'momentum':
                    0.86370430284520194, 'early_stopping': True, 'nesterovs_momentum': True},
                {'beta_1': 0.919473939313036, 'activation': 'relu', 'alpha': 1.6879174708893805, 'epsilon':
                    9.7045902021921481e-10, 'solver': 'sgd', 'power_t': 0.33021989249044037,
                    'learning_rate_init': 0.10898026569061127, 'hidden_layer_sizes': [955], 'max_iter': 164,
                    'beta_2': 0.99919431656457547, 'learning_rate': 'adaptive', 'momentum':
                    0.86370430284520194, 'early_stopping': True, 'nesterovs_momentum': True},
                {'max_iter': 350, 'beta_1': 0.93117401252301835, 'activation': 'tanh', 'epsilon': 1.1181028148969185e-08,
                    'beta_2': 0.99920214088237735, 'solver': 'sgd', 'early_stopping': True,
                    'nesterovs_momentum': False, 'momentum': 0.93188725719500154, 'alpha':
                    0.98570035796112321, 'learning_rate_init': 0.20724074137333875, 'power_t':
                    0.037268324764610727, 'learning_rate': 'adaptive', 'hidden_layer_sizes': [896, 679]},
                {'beta_1': 0.93117401252301835, 'activation': 'tanh', 'epsilon': 1.1181028148969185e-08,
                    'beta_2': 0.99920214088237735, 'solver': 'sgd', 'early_stopping': True,
                    'nesterovs_momentum': False, 'momentum': 0.93188725719500154, 'alpha':
                    0.98570035796112321, 'learning_rate_init': 0.20724074137333875, 'power_t':
                    0.037268324764610727, 'learning_rate': 'adaptive', 'hidden_layer_sizes': [896, 679]},
            ],
            LIN: [
                {'alpha': 0.0012311722512335377, 'eta0': 0.00031685190815167104, 'max_iter': 1000, 'power_t':
                    0.1840171707888663, 'learning_rate': 'invscaling', 'penalty': 'l1', 'n_jobs': -1,
                    'l1_ratio': 0.40558624196055393, 'random_state': 101, 'loss': 'log', 'tol': 1e-05},
            ],
            RF: [
                {'n_estimators': 47, 'max_depth': 10, 'max_leaf_nodes': 500, 'min_impurity_decrease':
                    0.00025947120035531014, 'min_samples_leaf': 2, 'min_weight_fraction_leaf':
                    0.00012438542624450296, 'min_samples_split': 3, 'criterion': 'entropy'},
            ],
            DT: [
                {'min_samples_split': 2, 'max_features': None, 'random_state': 101, 'min_samples_leaf': 2,
                    'min_weight_fraction_leaf': 0.0020780796100258966, 'criterion': 'entropy',
                    'min_impurity_decrease': 0.0012257673197162205, 'max_depth': 10, 'splitter': 'best',
                    'max_leaf_nodes': None},
            ],
            KNN: [
                {'n_neighbors': 12, 'metric': 'canberra', 'p': 1, 'weights': 'distance'},
                {'n_neighbors': 6, 'p': 2, 'metric': 'minkowski', 'weights': 'distance'},
            ],
        },
        SITE: {
            GCNN: [
                {'dropout': 0.7030699730394745, 'regularization': 0.0056770708415112515, 'Fs': [[9]], 'ps':
                    [[2]], 'momentum': 0.9723296129259393, 'M': [12], 'learning_rate': 0.004533461624908068,
                    'decay_rate': 0.9872381842057963, 'num_epochs': 200, 'decay_steps': 398, 'pool': 'mpool1',
                    'batch_size': 139, 'Ks': [[22]]},
            ],
            FFANN: [
                {'batch_size': 139, 'learning_rate_init': 0.0553037462469582, 'activation': 'relu', 'momentum':
                    0.8670995345995152, 'epsilon': 8.57966387882825e-09, 'nesterovs_momentum': True,
                    'learning_rate': 'invscaling', 'beta_2': 0.9985546737363457, 'power_t':
                    0.2260141186252257, 'early_stopping': False, 'beta_1': 0.864394587085052,
                    'hidden_layer_sizes': [997], 'solver': 'sgd', 'alpha': 0.8200640684863673},
                {'learning_rate_init': 0.0553037462469582, 'activation': 'relu', 'momentum':
                    0.8670995345995152, 'epsilon': 8.57966387882825e-09, 'nesterovs_momentum': True,
                    'learning_rate': 'invscaling', 'beta_2': 0.9985546737363457, 'power_t':
                    0.2260141186252257, 'early_stopping': False, 'beta_1': 0.864394587085052,
                    'hidden_layer_sizes': [997], 'solver': 'sgd', 'alpha': 0.8200640684863673},
                {'max_iter': 350, 'beta_1': 0.86439458708505201, 'learning_rate_init': 0.055303746246958198,
                    'hidden_layer_sizes': [997], 'solver': 'sgd', 'momentum': 0.86709953459951516,
                    'nesterovs_momentum': True, 'power_t': 0.22601411862522569, 'beta_2': 0.99855467373634565,
                    'learning_rate': 'invscaling', 'alpha': 0.82006406848636726, 'epsilon':
                    8.5796638788282495e-09, 'early_stopping': False, 'activation': 'relu'},
                {'beta_1': 0.86439458708505201, 'learning_rate_init': 0.055303746246958198,
                    'hidden_layer_sizes': [997], 'solver': 'sgd', 'momentum': 0.86709953459951516,
                    'nesterovs_momentum': True, 'power_t': 0.22601411862522569, 'beta_2': 0.99855467373634565,
                    'learning_rate': 'invscaling', 'alpha': 0.82006406848636726, 'epsilon':
                    8.5796638788282495e-09, 'early_stopping': False, 'activation': 'relu'},
            ],
            LIN: [
                {'penalty': 'l2', 'power_t': 0.085069721278429558, 'l1_ratio': 0.78472417245453174,
                    'learning_rate': 'invscaling', 'loss': 'modified_huber', 'alpha': 0.49413259295939205,
                    'eta0': 2.9915898133196991e-06},
            ],
            RF: [
                {'min_weight_fraction_leaf': 6.0054776997322797e-05, 'min_samples_leaf': 2, 'n_estimators':
                    53, 'criterion': 'entropy', 'min_samples_split': 2, 'max_depth': None,
                    'min_impurity_decrease': 0.00036814287906093349, 'max_leaf_nodes': None},
            ],
            DT: [
                {'criterion': 'gini', 'min_impurity_decrease': 7.7269701361130043e-05, 'splitter': 'best',
                    'max_features': 250, 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'max_depth': 100,
                    'min_weight_fraction_leaf': 0.0025284330944410848, 'min_samples_split': 2},
            ],
            KNN: [
                {'n_neighbors': 11, 'weights': 'uniform', 'p': 1, 'metric': 'canberra'},
            ],
        },
        SUBTYPE: {
            GCNN: [
                {'dropout': 0.4542127808744735, 'regularization': 0.005416198768028686, 'Fs': [[43]], 'ps':
                    [[2]], 'momentum': 0.972982486521958, 'M': [150, 150, 14], 'learning_rate':
                    0.0029527675535869005, 'decay_rate': 0.9761903148334647, 'num_epochs': 300, 'decay_steps':
                    362, 'pool': 'mpool1', 'batch_size': 88, 'Ks': [[8]]},
            ],
            FFANN: [
                {'activation': 'relu', 'nesterovs_momentum': True, 'solver': 'sgd', 'hidden_layer_sizes':
                    [997], 'learning_rate': 'invscaling', 'early_stopping': False, 'epsilon':
                    8.57966387882825e-09, 'beta_2': 0.9985546737363457, 'learning_rate_init':
                    0.0553037462469582, 'power_t': 0.2260141186252257, 'alpha': 0.8200640684863673,
                    'momentum': 0.8670995345995152, 'beta_1': 0.864394587085052},
                {'learning_rate': 'adaptive', 'beta_2': 0.99927254080997729, 'activation': 'logistic',
                    'beta_1': 0.93453259091631535, 'hidden_layer_sizes': [975], 'solver': 'sgd', 'power_t':
                    0.74385471896258193, 'early_stopping': True, 'epsilon': 1.206153958542538e-08, 'momentum':
                    0.90981549087918234, 'learning_rate_init': 0.37686906743651638, 'nesterovs_momentum':
                    True, 'alpha': 0.19856153391293785},
            ],
            LIN: [
                {'penalty': 'l2', 'power_t': 0.11388752410619822, 'l1_ratio': 0.73663692366676514, 'eta0':
                    9.9128007993276604e-05, 'learning_rate': 'invscaling', 'alpha': 0.36291034438506586,
                    'loss': 'modified_huber'},
            ],
            RF: [
                {'criterion': 'gini', 'min_samples_split': 2, 'min_weight_fraction_leaf':
                    0.002486914559785937, 'min_impurity_decrease': 9.9073092561388911e-06, 'n_estimators':
                    436, 'min_samples_leaf': 1, 'max_depth': 100, 'max_leaf_nodes': 100},
            ],
            DT: [
                {'splitter': 'best', 'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 1,
                    'min_weight_fraction_leaf': 0.011206544817711744, 'max_depth': 5, 'min_samples_split': 2,
                    'max_leaf_nodes': 100, 'min_impurity_decrease': 3.1882056516088607e-05},
            ],
            KNN: [
                {'n_neighbors': 9, 'weights': 'uniform', 'metric': 'canberra', 'p': 9},
            ],
        },
    },
    GSE92742_PROSTATE: {
        MOA_PROSTATE: {
            GCNN: [
                {'dropout': 0.5, 'regularization': 0.004, 'Fs': [[25]], 'ps': [[2]], 'momentum': 0.97, 'M':
                    [168, 14, 9], 'learning_rate': 0.005, 'decay_rate': 0.95, 'num_epochs': 200,
                    'decay_steps': 415, 'pool': 'mpool1', 'batch_size': 55, 'Ks': [[15]]},
            ],
            FFANN: [
                {'learning_rate_init': 0.0553037462469582, 'max_iter': 350, 'beta_2': 0.9985546737363457,
                    'early_stopping': False, 'activation': 'relu', 'momentum': 0.8670995345995152, 'beta_1':
                    0.864394587085052, 'learning_rate': 'invscaling', 'nesterovs_momentum': True,
                    'hidden_layer_sizes': [997], 'alpha': 0.8200640684863673, 'power_t': 0.2260141186252257,
                    'solver': 'sgd', 'epsilon': 8.57966387882825e-09},
            ],
            LIN: [
                {'alpha': 0.0012311722512335377, 'eta0': 0.00031685190815167104, 'max_iter': 1000, 'power_t':
                    0.1840171707888663, 'learning_rate': 'invscaling', 'penalty': 'l1', 'n_jobs': -1,
                    'l1_ratio': 0.40558624196055393, 'random_state': 101, 'loss': 'log', 'tol': 1e-05},
            ],
            RF: [
                {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 25, 'min_weight_fraction_leaf':
                    0.00043290898297300624, 'min_impurity_decrease': 3.6379730587986808e-05, 'max_leaf_nodes':
                    500, 'n_estimators': 411, 'criterion': 'gini'},
            ],
            DT: [
                {'max_features': 250, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0014646397820345471,
                    'max_leaf_nodes': None, 'splitter': 'best', 'min_samples_split': 2,
                    'min_weight_fraction_leaf': 0.00018089997347969574, 'criterion': 'entropy', 'max_depth':
                    25},
            ],
            KNN: [
                {'n_neighbors': 13, 'metric': 'canberra', 'p': 12, 'weights': 'distance'},
            ],
        },
    },
}
