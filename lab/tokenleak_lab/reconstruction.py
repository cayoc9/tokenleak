"""API do atacante: recebe somente observacoes, vocabulario publico e perfis."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def topk(scores, token_ids, k=5):
    """Ranking deterministico, incluindo empates na fronteira do Top-k."""
    scores = np.asarray(scores)
    k = min(k, scores.shape[1])
    indices = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    threshold = np.take_along_axis(scores, indices, axis=1).min(axis=1)
    for row in range(len(scores)):
        greater = np.flatnonzero(scores[row] > threshold[row])
        tied = np.flatnonzero(scores[row] == threshold[row])
        if len(tied) > k - len(greater):
            tied = tied[np.argsort(token_ids[tied], kind='stable')][:k-len(greater)]
            indices[row] = np.concatenate([greater, tied])
    values = np.take_along_axis(scores, indices, axis=1)
    order = np.lexsort((token_ids[indices], -values), axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    values = np.take_along_axis(scores, indices, axis=1)
    predictions = token_ids[indices].copy()
    predictions[~np.isfinite(values)] = -1
    return predictions, values


class Reconstructor:
    def __init__(self, profiles):
        self.profiles = profiles
        self.normalized = profiles.cache / np.maximum(np.linalg.norm(profiles.cache, axis=1)[:, None], 1e-15)

    def features(self, observations):
        timing = np.exp(-0.5 * ((observations.timing[:, None] - self.profiles.mu) / self.profiles.sigma) ** 2)
        normalized = observations.cache.astype(float)
        normalized /= np.maximum(np.linalg.norm(normalized, axis=1)[:, None], 1e-15)
        return timing, normalized @ self.normalized.T

    def rank_features(self, timing, cache, weights, observed_timing=None, prune_sigma=None):
        scores = (weights[0] * timing + weights[1] * cache) / sum(weights)
        fallback = np.zeros(len(scores), dtype=bool)
        counts = np.full(len(scores), len(self.profiles.token_ids), dtype=np.int32)
        if prune_sigma is not None:
            allowed = np.abs(observed_timing[:, None] - self.profiles.mu) <= prune_sigma * self.profiles.sigma
            fallback = ~allowed.any(axis=1)
            allowed[fallback] = True
            counts = allowed.sum(axis=1).astype(np.int32)
            scores[~allowed] = -np.inf
        predicted, values = topk(scores, self.profiles.token_ids)
        return predicted, values[:, 0], counts, fallback


def fit_weights(profiles, observations, labels, cfg, rng):
    """Apenas dados de perfilamento: pares corretos, aleatorios e proximos no timing."""
    truth_index = np.searchsorted(profiles.token_ids, labels)
    n = len(labels)
    uniform_count = cfg['uniform_negatives_per_position']
    uniform = rng.integers(0, len(profiles.token_ids), (n, uniform_count))
    while np.any(uniform == truth_index[:, None]):
        bad = uniform == truth_index[:, None]
        uniform[bad] = rng.integers(0, len(profiles.token_ids), int(bad.sum()))
    ordered = np.argsort(profiles.mu)
    centers = np.searchsorted(profiles.mu[ordered], observations.timing)
    hard_count = cfg['hard_timing_negatives_per_position']
    hard = np.empty((n, hard_count), dtype=int)
    for row in range(n):
        neighborhood = ordered[max(0, centers[row]-hard_count-2):min(len(ordered), centers[row]+hard_count+3)]
        neighborhood = neighborhood[neighborhood != truth_index[row]]
        hard[row] = neighborhood[np.argsort(np.abs(profiles.mu[neighborhood] - observations.timing[row]), kind='stable')[:hard_count]]
    candidates = np.column_stack([truth_index, uniform, hard])
    phi_t = np.exp(-0.5 * ((observations.timing[:, None] - profiles.mu[candidates]) / profiles.sigma[candidates]) ** 2)
    obs = observations.cache.astype(float)
    obs /= np.maximum(np.linalg.norm(obs, axis=1)[:, None], 1e-15)
    references = profiles.cache[candidates]
    phi_c = np.einsum('nj,nkj->nk', obs, references) / np.maximum(np.linalg.norm(references, axis=2), 1e-15)
    x = np.stack([phi_t, phi_c], axis=-1).reshape(-1, 2)
    y = np.zeros(candidates.shape); y[:, 0] = 1; y = y.ravel()
    sample_weight = np.where(y == 1, 0.5/n, 0.5/(len(y)-n))
    penalty = cfg['nonnegative_logistic_l2']

    def objective(beta):
        z = x @ beta[:2] + beta[2]
        loss = np.dot(sample_weight, np.logaddexp(0, z) - y*z) + 0.5*penalty*np.dot(beta[:2], beta[:2])
        residual = sample_weight * (expit(z) - y)
        gradient = np.r_[x.T @ residual + penalty*beta[:2], residual.sum()]
        return loss, gradient

    result = minimize(objective, [1., 1., -1.], jac=True, method='L-BFGS-B',
                      bounds=[(0, None), (0, None), (None, None)], options={'maxiter': 500, 'ftol': 1e-12})
    if not result.success or result.x[:2].sum() <= 0:
        raise ValueError(f'Weight optimization failed: {result.message}')
    weights = result.x[:2] / result.x[:2].sum()
    metadata = {'weights': weights.tolist(), 'coefficients': result.x[:2].tolist(),
                'intercept': float(result.x[2]), 'loss': float(result.fun), 'iterations': int(result.nit),
                'pair_count': len(y), 'positive_count': n,
                'method': 'class-balanced nonnegative logistic loss with intercept and fixed L2; normalized coefficients used by TRCS',
                'labels_used': 'training partition only', 'not_a_calibrated_TRCS_probability': True}
    return weights, metadata, profiles.token_ids[candidates]
