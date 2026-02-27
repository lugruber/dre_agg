import numpy
import numpy as np
from scipy.stats import norm, multivariate_normal
from operator import itemgetter
from copy import deepcopy


class GMDClassif:

    def __init__(self, nsamples, comps, comp_weights):
        super(GMDClassif).__init__()
        self.comps = list(comps)
        self.comp_weights = comp_weights
        self.nsamples = nsamples
        self.ncomps = self.comp_weights.shape[0]
        ncomps_dist1 = np.random.randint(low=1, high=4)
        self.comps_dist1 = np.random.choice(a=self.ncomps, size=ncomps_dist1, replace=False)

        self.comps_dist2 = np.setdiff1d(np.arange(self.ncomps), self.comps_dist1)
        self.comp_weights[self.comps_dist1] /= self.comp_weights[self.comps_dist1].sum()
        self.comp_weights[self.comps_dist2] /= self.comp_weights[self.comps_dist2].sum()

    def sample(self):
        samples_per_comp = (self.nsamples / 2 * self.comp_weights).astype(int)
        samples = []
        dim = self.comps[0][0].shape[0]
        for i in range(self.ncomps):
            mu, cov = self.comps[i][0], self.comps[i][1]
            if dim > 1:
                samples.append(multivariate_normal.rvs(mean=mu, cov=cov, size=samples_per_comp[i]))
            else:
                samples.append(norm.rvs(loc=mu, scale=cov, size=samples_per_comp[i]))
        if dim == 1:
            for i in range(len(samples)):
                samples[i] = samples[i].reshape(-1, 1)
        if self.comps_dist1.shape[0] > 1:
            self.dist1 = np.concatenate(itemgetter(*list(self.comps_dist1))(samples), axis=0)
        else:
            self.dist1 = itemgetter(*list(self.comps_dist1))(samples)
        np.random.shuffle(self.dist1)
        self.rest = np.setdiff1d(np.arange(self.ncomps), self.comps_dist1)
        if self.rest.shape[0] > 1:
            self.dist2 = np.concatenate(itemgetter(*list(self.rest))(samples), axis=0)
        else:
            self.dist2 = itemgetter(*list(self.rest))(samples)
        np.random.shuffle(self.dist2)
        return self.dist1, self.dist2

    def get_ratio(self, dset):
        dim = dset.shape[1]
        if dim > 1:
            if self.comps_dist1.shape[0] > 1:
                denom = self.comp_weights[self.comps_dist1].reshape(-1, 1) * numpy.stack(
                    [multivariate_normal.pdf(x=dset, mean=comp[0], cov=comp[1])
                     for comp in itemgetter(*list(self.comps_dist1))(self.comps)])
            else:
                denom = self.comp_weights[self.comps_dist1].reshape(-1, 1) * numpy.stack(
                    [multivariate_normal.pdf(x=dset, mean=comp[0], cov=comp[1])
                     for comp in [itemgetter(*list(self.comps_dist1))(self.comps)]])
            if self.rest.shape[0] > 1:
                nom = self.comp_weights[self.rest].reshape(-1, 1) * numpy.stack(
                    [multivariate_normal.pdf(x=dset, mean=comp[0], cov=comp[1])
                     for comp in itemgetter(*list(self.rest))(self.comps)])
            else:
                nom = self.comp_weights[self.rest].reshape(-1, 1) * numpy.stack(
                    [multivariate_normal.pdf(x=dset, mean=comp[0], cov=comp[1])
                     for comp in [itemgetter(*list(self.rest))(self.comps)]])
        else:
            dset = dset.squeeze(axis=-1)
            if self.comps_dist1.shape[0] > 1:
                denom = self.comp_weights[self.comps_dist1].reshape(-1, 1) * numpy.stack(
                    [norm.pdf(x=dset, loc=comp[0], scale=comp[1])
                     for comp in itemgetter(*list(self.comps_dist1))(self.comps)])
            else:
                denom = self.comp_weights[self.comps_dist1].reshape(-1, 1) * numpy.stack(
                    [norm.pdf(x=dset, loc=comp[0], scale=comp[1])
                     for comp in [itemgetter(*list(self.comps_dist1))(self.comps)]])
            if self.rest.shape[0] > 1:
                nom = self.comp_weights[self.rest].reshape(-1, 1) * numpy.stack(
                    [norm.pdf(x=dset, loc=comp[0], scale=comp[1])
                     for comp in itemgetter(*list(self.rest))(self.comps)])
            else:
                nom = self.comp_weights[self.rest].reshape(-1, 1) * numpy.stack(
                    [norm.pdf(x=dset, loc=comp[0], scale=comp[1])
                     for comp in [itemgetter(*list(self.rest))(self.comps)]])
        denom = denom.sum(axis=0)
        nom = nom.sum(axis=0)
        return nom / denom

