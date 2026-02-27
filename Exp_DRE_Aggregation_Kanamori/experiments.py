import sys
sys.path.append('../')
from Exp_DRE_Aggregation_Kanamori.data import GMDClassif
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import argparse
from scipy import stats
from Exp_DRE_Aggregation_Kanamori.models_torch import IterLogi, IterSquare, IterCpeExp, IteratedKulsif, METHOD_DICT
from Exp_DRE_Aggregation_Kanamori.models import nmse
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

arg_parser = argparse.ArgumentParser(description=r'Set parameters for DRE')
arg_parser.add_argument(r'--dataset', type=str, help=r'name of the dataset', required=True)
arg_parser.add_argument(r'--method', type=str, help=r'name of the dre method', required=True)
arg_parser.add_argument(r'--agg', action='store_true', help=r'perform aggregation')
arg_parser.add_argument(r'--bma', action='store_true', help=r'perform bayesian model averaging as aggregation')
arg_parser.add_argument(r'--super', action='store_true', help=r'use super learner method as aggregation')
arg_parser.add_argument(r'--rcond', type=float, help=r'condition number for pseudo inverse', default=1e-3)
arg_parser.add_argument(r'--seed', type=int,help=r'seed used for experiment', default=1)
arg_parser.add_argument(r'--nreplicates', type=int,help=r'number of geometric datasets', default=10)
arg_parser.add_argument(r'--nsamples', type=int,help=r'number of samples', default=5000)
arg_parser.add_argument(r'--ncomps', type=int,help=r'number of components', default=4)
arg_parser.add_argument(r'--dim', type=int,help=r'dimension of probability space', default=50)
arg_parser.add_argument(r'--kernel', type=str, help=r'type of kernel', default='rbf')
args = arg_parser.parse_args()

np.random.seed(args.seed)
if args.kernel =='rbf':
    kernel = rbf_kernel

int_start, int_end = -0.1, 1
best_scores = np.zeros(shape=(args.nreplicates, 10))
param_grid = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=6, dtype=float)))}
# outer loop over different datasets    
for rep in range(args.nreplicates):
    comp_means = np.random.uniform(low=int_start + 0.3, high=int_end - 0.3, size=(args.ncomps, args.dim))
    comp_stds = np.random.uniform(low=0.7, high=1., size=(args.ncomps, args.dim))
    if args.dim > 1:
	eigs = np.random.uniform(low=1.5, high=2.5, size=(args.ncomps, args.dim))
	eigs = eigs / eigs.sum(axis=1).reshape(-1, 1) * eigs.shape[1]
	comp_covmats = [np.diag(comp_std) @ stats.random_correlation.rvs(eigs=eig) @ np.diag(comp_std)
	                for comp_std, eig in zip(comp_stds, eigs)]
    else:
	# square of this will be variance of random variables
	comp_covmats = comp_stds
    comp_weights = np.random.uniform(low=0.1, high=1., size=args.ncomps)
    comp_weights /= comp_weights.sum()
    dataset = GMDClassif(nsamples=args.nsamples, comps=zip(comp_means, comp_covmats), comp_weights=comp_weights)
    for s in range(10):
	source_X, target_X = dataset.sample()
	# source gets pseudo-label y=1, target y=0, but in Bregman loss source:y=-1, target: y=1
	source_Xy = np.concatenate((source_X, np.ones(source_X.shape[0]).reshape(-1, 1)), axis=1)
	target_Xy = np.concatenate((target_X, np.zeros(target_X.shape[0]).reshape(-1, 1)), axis=1)
	data_Xy = np.concatenate((source_Xy, target_Xy), axis=0)
	np.random.shuffle(data_Xy)
	dens_rats = dataset.get_ratio(dset=data_Xy[:, :-1])
	data_Xy_train, data_Xy_test, dens_rats_train, dens_rats_test = train_test_split(
		data_Xy, dens_rats, test_size=0.2, shuffle=True, stratify=data_Xy[:, -1])
	distances = np.triu(cdist(XA=data_Xy_train[:, :-1], XB=data_Xy_train[:, :-1]))
        med = np.median(distances[distances > 0])
        rbf_gam = 1 / med
        if args.agg == False:
            train_idx_h, val_idx_h = train_test_split(np.arange(data_Xy_train[:, :-1].shape[0]),
                                                      test_size=0.2, stratify=data_Xy_train[:, -1], shuffle=True)
            est = METHOD_DICT[args.method](kernel=rbf_kernel, rbf_gam=rbf_gam)
            clf = GridSearchCV(estimator=est, param_grid=param_grid,
                                   cv=[(train_idx_h, val_idx_h)],
                                   scoring='neg_root_mean_squared_error',
                                   )
            clf.fit(X=data_Xy_train, y=dens_rats_train)
            preds = best_model.predict(data_Xy_test)
            best_scores[rep, s] = nmse(y_true=dens_rats_test, y_pred=preds)
        else:
            estimator_weights = []
            preds_train = []
            preds_test = []
            for par in param_grid['reg_par']:
                if args.method != 'iteratedkulsif':
                    estimator = METHOD_DICT[args.method](kernel=rbf_kernel, rbf_gam=rbf_gam, reg_par=par)
                else:
                    estimator = METHOD_DICT[args.method](input_size=data_Xy_train.shape[1] - 1, reg_par=par)
                estimator.fit(X=data_Xy_train, y=dens_rats_train)
                estimator_weights.append(estimator.get_paras())
                preds_train.append(estimator.predict(data_Xy_train, return_f=True))
                preds_test.append(estimator.predict(data_Xy_test, return_f=False))
            
            preds = np.stack(arrays=preds_train, axis=0)
            if args.method == 'iteratedkulsif':
                estimator_weights = np.squeeze(np.stack(arrays=estimator_weights, axis=0), axis=-1)
            else:
                estimator_weights = np.stack(arrays=estimator_weights, axis=0)
            if args.bma:
                rss = ((dens_rats - preds) ** 2).sum(axis=1)
                n, k = len(dens_rats_train), data_Xy_train.shape[1] - 1
                bics = n * np.log(rss / n) + k * np.log(n)
                min_bic = np.min(bics)
                weights = np.exp(-0.5 * (bics - min_bic))
                posterior_probs = weights / np.sum(weights)
                agg_preds = (np.stack(arrays=preds_test, axis=0) * posterior_probs.reshape(-1, 1)).sum(axis=0)
            elif args.super:
                def superlearner_loss(weights, predictions, labels):
                    epsilon = 1e-8
                    ensemble_preds = (predictions * weights.reshape(-1, 1)).sum(axis=0)
                    term1 = -1 * (labels == 0) * np.log(ensemble_preds + epsilon)
                    term2 = (labels == 1) * np.log(ensemble_preds + epsilon)
                    return np.mean(term1 + term2)

                n_learners, n_samples = preds.shape
                initial_weights = (np.ones(n_learners) / n_learners)
                bounds = [(0, 1) for _ in range(n_learners)]
                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                weights = minimize(
                    superlearner_loss,
                    initial_weights,
                    args=(preds, data_Xy_train[:, -1]),
                    bounds=bounds,
                    constraints=constraints,
                    method='SLSQP')
                agg_preds = (np.stack(arrays=preds_test, axis=0) * weights.reshape(-1, 1)).sum(axis=0)
            else:
                gram_source = (preds[:, np.newaxis, data_Xy_train[:, -1] == 1] * preds[np.newaxis, :, data_Xy_train[:, -1] == 1] * (1 -
                    preds[np.newaxis, 0, data_Xy_train[:,-1] == 1]) ** (-2)).mean(axis=-1)
                gram_target = (preds[:, np.newaxis, data_Xy_train[:, -1] == 0] * preds[np.newaxis, :, data_Xy_train[:, -1] == 0] * preds[
                    np.newaxis, 0, data_Xy_train[:,-1] == 0] ** (-2)).mean(axis=-1)
                gram = gram_source + gram_target
                source_X, target_X = data_Xy_train[data_Xy_train[:, -1] == 1][:, :-1], data_Xy_train[data_Xy_train[:, -1] == 0][:, :-1]
                if args.method == 'iteratedkulsif':
                    kern_mat = 1 + rbf_kernel(X=np.concatenate((source_X, target_X), axis=0), gamma=rbf_gam)
                else:
                    kern_mat = 1 + rbf_kernel(X=np.concatenate((target_X, source_X), axis=0), gamma=rbf_gam)
                reg = param_grid['reg_par'][0] * estimator_weights @ kern_mat @ estimator_weights.T
                gram += reg
            
                # compute pseudo-inverse and use condition number
                gram_inv = np.linalg.pinv(gram, rcond=args.rcond)
                # source gets pseudo-label y=1, target y=0, but in Bregman loss source:y=-1, target: y=1
                # here different sign than in gram because pseudo-labels are used in inner product
                rhs = (preds[:, data_Xy_train[:, -1] == 0] * preds[0, data_Xy_train[:, -1] == 0].reshape(1, -1) ** (-2)).mean(axis=-1) - (
                       preds[:, data_Xy_train[:, -1] == 1] * (1 - preds[0, data_Xy_train[:, -1] == 1].reshape(1, -1)) ** (-2)).mean(axis=-1)
                rhs += reg.diagonal()
                agg_weights = gram_inv @ rhs
                agg_preds = (np.stack(arrays=preds_test, axis=0) * agg_weights.reshape(-1, 1)).sum(axis=0)
            
            best_scores[rep, s] = nmse(y_true=dens_rats_test, y_pred=agg_preds)
np.save(file=f'./results/results_{args.dataset}_{args.method}_agg{args.agg}', arr=best_scores)

