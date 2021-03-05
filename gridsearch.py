import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,\
    recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from itertools import combinations
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csc_matrix

class GridSearch:
    def __init__(self):
        self.data = None
        self.clf_dict = None
        self.clf_hypers = None
        self.metric = None
        self._best = None
        self.results = None
        self.target = None


    def _plot_metric(self, metric='auc', save=False):
        """
        Plots classifer performance for all classifiers for given metric.
        \nmetric: string metric
        \nsave: boolean control to save plots or not
        """
        grid_results = self.results
        metric = metric
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for clf in grid_results:
            if clf == 'best_overall':
                continue
            clf_iterations = grid_results[clf].get('iterations')
            x = [r.get('id') for r in clf_iterations]
            y = [r.get(metric) for r in clf_iterations]
            line, = ax.plot(x, y, label=clf)

        plt.title(metric.capitalize()+" Comparison")
        plt.xlabel('Model ID')
        plt.ylabel('Percent Score')
        ax.legend()

        if save:
            try:
                plt.savefig('./GridSearch/'+metric+'.png', format='png', dpi=400)
        
            except(FileNotFoundError):
                from os import mkdir
                mkdir('GridSearch')
                plt.savefig('./GridSearch/'+metric+'.png', format='png', dpi=400)


    def _get_hypers(self, clf_params):
        if not self.clf_hypers:
            raise AttributeError('Hyper-parameters have not been set. \
                Run set_params() first.')

        gridlist = []

        # Get all possible combinations for clf_params:
        for parameter in clf_params:
            value_list = clf_params.get(parameter)

            for value in value_list:
                item = {parameter: value}
                gridlist.append(item)

        paramset = combinations(gridlist, len(clf_params.keys()))

        # Get unique params per iteration:
        trimmed_list = []
        
        for parameter in paramset:
            seen_params = []
            param_dict = {}
            keep = True
            for s in parameter:
                if s.keys() in seen_params:
                    keep = False
                    break

                param_dict.update(s)
                seen_params.append(s.keys())

            if keep:
                trimmed_list.append(param_dict)

        return trimmed_list

    def npv_score(self, ytest, yhat):
        # Negative Predictive Value identifies the probability that a patient is actually negative for a test
        confusion = confusion_matrix(yhat, ytest)
        TN = confusion[0, 0]
        FN = confusion[1, 0]
        # convert to float and Calculate score 
        npv = (TN.astype(float)/(FN.astype(float)+TN.astype(float)))
        #    (True Negative/(Predicted Negative + True Negative))
        return npv

    def ppv_score(self, ytest, yhat):
        # Posotive Predictive Value identifies the probability that a patient is actually positive for a test
        confusion = confusion_matrix(yhat, ytest)
        FP = confusion[0, 1]
        TP = confusion[1, 1]
        # convert to float and Calculate score 
        ppv = (TP.astype(float)/(FP.astype(float)+TP.astype(float)))
        #    (True Negative/(Predicted Negative + True Positive))
        return ppv  

    def specificity_score(self, ytest, yhat):
        confusion = confusion_matrix(yhat, ytest)
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        spec = TN.astype(float)/(FP.astype(float) + TN.astype(float))
        return spec

    def set_params(self, data, classifiers={}, param_grid={}, metric='auc'):
        """
        \ndata: packed variables for explanatory data, response
        \nclassifiers: dictionary of classifier_name:classifier_object
        \nparam_grid: dictionary of classifier dictionaries with list of params
        \nmetric: string of metric to use to find best model.
            \nSupports: accuracy, precision, recall, f1_score, auc
        """
        self.clf_hypers = param_grid
        self.data = data
        self.clf_dict = classifiers
        self.metric = metric


    def run_grid(self, n_folds=5, splits=True, scale=True, sparse=False, \
        verbose=False, target=''):
        """
        Runs a grid search for each classifier and each iteration of paramsets
        \nn_folds: number of kfolds for stratified cross validation
        \nsplits: boolean for returning split dictionary (much more verbose)
        \nscale: boolean to use min_max scaler on input data
        \nReturns: dictionary
        """
        import logging
        try:
            logging.basicConfig(filename='./GridSearch/models.log', level=logging.INFO)
        except FileNotFoundError:
            from os import mkdir
            mkdir('./GridSearch')
            logging.basicConfig(filename='./GridSearch/models.log', level=logging.INFO)

        print('Grid Search running...')
        clf_dict = self.clf_dict
        clf_hypers = self.clf_hypers
        metric = self.metric

        M, L = self.data # unpack data container
        if sparse:
            M = csc_matrix(M)
   
        scaler = MaxAbsScaler()
        if verbose:
            print('scaling data...')
        scaled_data = scaler.fit_transform(M)
        if scale:
            M = scaled_data
        
        kf = StratifiedKFold(n_splits=n_folds)
        grid_dict = {
            'best_overall': {
                metric: 0}}  # classic explication of results

        def split_avg(metric, split_results):
            metrics = [split_results[id][metric] for id in split_results]
            return np.mean(metrics)
        
        for a_clf in clf_dict:  # for each classifier:
            # initialize dictionarys
            clf_results = []
            iteration_results = {}
            split_results = {}
            best_iteration = {metric: 0}

            # for each parameter setting:
            clf_iterations = self._get_hypers(clf_hypers[a_clf])
            for model_id, iteration in enumerate(clf_iterations):
                if verbose:
                    print('\n\ntraining '+a_clf+' model_id: '+str(model_id))
                    print('--params: '+str(iteration))
                    logging.info('\n\ntraining '+a_clf+' model_id: '+str(model_id))
                    logging.info('--params: '+str(iteration))


                clf_params = iteration
                clf = None

                for split_id, (train_index, test_index) \
                        in enumerate(kf.split(M, L)):  # for each kfold
                    clf = clf_dict[a_clf](**clf_params)  # unpack parameters

                    try:  # always set random and all processors
                        clf.set_params(random_state=86)
                        clf.set_params(n_jobs=-1)
                    except ValueError:
                        pass  # skip incompatible params

                    clf.fit(M[train_index], L[train_index])
                    
                    preds = clf.predict(M[test_index])
                    if verbose:
                        print('----k_fold '+str(split_id+1)+' complete')
                        logging.info('----k_fold '+str(split_id+1)+' complete')

                    split_results[split_id] = {
                        'train_index': train_index,
                        'test_index': test_index,
                        'split_accuracy': accuracy_score(
                            L[test_index], preds),
                        'split_precision': precision_score(
                            L[test_index], preds),
                        'split_recall': recall_score(
                            L[test_index], preds),
                        'split_f1_score': f1_score(
                            L[test_index], preds),
                        'split_auc': roc_auc_score(
                            L[test_index], preds),
                        'split_specificity': self.specificity_score(
                            L[test_index], preds),
                        'split_ppv': self.ppv_score(
                            L[test_index], preds),
                        'split_npv': self.npv_score(
                            L[test_index], preds),                            
                        }

                iteration_results = {
                    'id': str(model_id),
                    'type': a_clf,
                    'clf': clf,
                    'set_params': clf_params,
                    'splits': split_results,
                    'accuracy': split_avg('split_accuracy', split_results),
                    'precision': split_avg('split_precision', split_results),
                    'recall': split_avg('split_recall', split_results),
                    'f1_score': split_avg('split_f1_score', split_results),
                    'auc': split_avg('split_auc', split_results),
                    'specificity': split_avg('split_specificity', split_results),
                    'ppv': split_avg('split_ppv', split_results),
                    'npv': split_avg('split_npv', split_results)
                    }

                if not splits:
                    del iteration_results['splits']

                if verbose:
                    print(iteration_results)
                    logging.info(iteration_results)
        
                clf_results.append(iteration_results)
                
                if iteration_results[metric] > best_iteration[metric]:
                    best_iteration = iteration_results

            grid_dict[a_clf] = {
                'iterations': clf_results,
                'best_'+a_clf: best_iteration
                }

            if best_iteration[metric] > grid_dict['best_overall'][metric]:
                grid_dict['best_overall'] = best_iteration

        print('Grid Search complete!\n\nBest Model:')
        print(grid_dict['best_overall'])
        logging.info('Grid Search complete!\n\nBest Model:')
        logging.info(grid_dict['best_overall'])

        self._best = grid_dict['best_overall']['clf']
        self.results = grid_dict
        return self.results

    def plot_metrics(self, save=False):
        metrics = ['accuracy', 'auc', 'f1_score', 'precision', 'recall', 'specificity', 'ppv','npv']
        for metric in metrics:
            self._plot_metric(metric, save)
        if save:
            print('\nPlots Saved!')

    def update_best_metric(results:dict, metric:str = 'npv'):
        for clf_key in results:
            if clf_key != 'best_overall':
                clf_dict = results[clf_key]
                best_clf =  clf_dict['best_'+clf_key]
                for iteration in clf_dict['iterations']:
                    if iteration[metric] > best_clf[metric]:
                        best_clf = iteration
                    if iteration[metric] > results['best_overall'][metric]:
                        results['best_overall'] = iteration   
        return results

def main():  # POC for the class

    # Get Data:
    bc_data = load_breast_cancer()

    # Set X,y:
    M = bc_data.data
    L = bc_data.target

    # Package data
    data = (M, L)

    # Choose classifiers to run
    classifiers = {
        'Random_Forest': RandomForestClassifier,
        'Logistic_Regression': LogisticRegression,
        'SVM': SVC
    }

    # Edit grid params.
    # Hint: run RandomForestClassifier().get_params() to get param list.
    param_grid = {
        'Random_Forest': {
            # 'bootstrap': True,
            # 'ccp_alpha': 0.0,
            'class_weight': ['balanced'],
            'criterion': ['gini', 'entropy'],
            # 'max_depth': None,
            'max_features': ['auto', None],
            # 'max_leaf_nodes': None,
            # 'max_samples': None,
            # 'min_impurity_decrease': 0.0,
            # 'min_impurity_split': None,
            # 'min_samples_leaf': 1,
            # 'min_samples_split': 2,
            # 'min_weight_fraction_leaf': 0.0,
            'n_estimators': [100]
            # 'n_jobs': None,
            # 'oob_score': False,
            # 'random_state': None,
            # 'verbose': 0,
            # 'warm_start': False
            },
        'Logistic_Regression': {
            'C': [1.0, .5],
            'class_weight': ['balanced'],
            # 'dual': False,
            # 'fit_intercept': True,
            'intercept_scaling': [1, 10],
            # 'l1_ratio': None,
            # 'max_iter': [10, 100],
            # 'multi_class': 'auto',
            # 'n_jobs': None,
            'penalty': ['l2'],
            # 'random_state': None,
            'solver': ['sag'] # sag needs scaled data, but runs well on large datasets
            # 'tol': 0.0001,
            # 'verbose': 0,
            # 'warm_start': False
            },
        'SVM': {
            'C': [.1, 1],
            # 'break_ties': False,
            # 'cache_size': 200,
            'class_weight': ['balanced'],
            # 'coef0': 0.0,
            # 'decision_function_shape': 'ovr',
            # 'degree': 3,
            # 'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
            # 'max_iter': -1,
            # 'probability': False,
            # 'random_state': None,
            # 'shrinking': True,
            # 'tol': 0.001,
            # 'verbose': False
            }
        }

    gs = GridSearch()
    gs.set_params(
        data=data,
        classifiers=classifiers,
        param_grid=param_grid,
        metric='auc'
    )
    results = gs.run_grid(n_folds=3, splits=False, scale=False, sparse=True)
    gs.plot_metrics(save=True)

    return results


if __name__ == "__main__":
    grid_results = main()
