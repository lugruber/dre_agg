import numpy as np
from scipy.optimize import minimize

def dre_aggregation(iwv_dict, hess_reg_par, hess_mod_id, rcond, sel_lambs=slice(None), bma=False, sup=False):
    preds = iwv_dict['preds_fy'][:-1] + 1e-20 #to not divide by 0 when computing norm below
    if sel_lambs is not None:
        preds = preds[sel_lambs]
    y = iwv_dict['preds_fy'][-1]
    source_cond = y == 1
    target_cond = y == 0
    if bma:
        rss = ((y - preds) ** 2).sum(axis=1)
        k, n = preds.shape
        bics = n * np.log(rss / n) + k * np.log(n)
        min_bic = np.min(bics)
        weights = np.exp(-0.5 * (bics - min_bic))
        agg_weights = weights / np.sum(weights)
    elif sup:
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
        agg_weights = minimize(
            superlearner_loss,
            initial_weights,
            args=(preds, y),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP')   
    else:     
        gram_source = (preds[:, np.newaxis, source_cond] * preds[np.newaxis, :, source_cond] * (1 -
            preds[np.newaxis, hess_mod_id, source_cond] + 1e-20) ** (-2)).mean(axis=-1)
        gram_target = (preds[:, np.newaxis, target_cond] * preds[np.newaxis, :, target_cond] * (preds[
            np.newaxis, hess_mod_id, target_cond] + 1e-20) ** (-2)).mean(axis=-1)
        gram = gram_source + gram_target
        kern = iwv_dict['weigh_kernmat']
        if sel_lambs is not None:
            kern = kern[sel_lambs, sel_lambs]
        reg = hess_reg_par * kern
        gram += reg
        gram_inv = np.linalg.pinv(gram, rcond=rcond)
        # broadcasting is automatically done correct, reshape not absolutely necessary, for gram not done either
        rhs = (preds[:, target_cond] * preds[hess_mod_id, target_cond].reshape(1, -1) ** (-2)).mean(axis=-1) - (
            preds[:, source_cond] * (1 - preds[hess_mod_id, source_cond].reshape(1, -1)) ** (-2)).mean(axis=-1)
        rhs += reg[hess_mod_id]
        agg_weights = gram_inv @ rhs
    # iterate over batches
    for i, s_pred in enumerate(iwv_dict['s_preds']):
        if sel_lambs is not None:
            s_pred = s_pred[sel_lambs]
        iwv_dict['s_preds'][i] = (s_pred * agg_weights.reshape(-1, 1)).sum(axis=0)
    iwv_dict['s_preds'] = np.concatenate(iwv_dict['s_preds'])
    for i, t_pred in enumerate(iwv_dict['t_preds']):
        if sel_lambs is not None:
            t_pred = t_pred[sel_lambs]
        iwv_dict['t_preds'][i] = (t_pred * agg_weights.reshape(-1, 1)).sum(axis=0)
    iwv_dict['t_preds'] = np.concatenate(iwv_dict['t_preds'])

    return iwv_dict
