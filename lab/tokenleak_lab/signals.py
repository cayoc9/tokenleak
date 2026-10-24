"""Ambiente sintetico separado do reconstrutor; perfis estimados por amostras."""
from dataclasses import dataclass
import numpy as np


def rng_for(seed, component):
    return np.random.default_rng(np.random.SeedSequence([20261023, int(seed), int(component)]))


@dataclass
class Observations:
    timing: np.ndarray
    cache: np.ndarray


@dataclass
class Profiles:
    token_ids: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    cache: np.ndarray


class SyntheticEnvironment:
    def __init__(self, vocabulary, cfg):
        self.tokens = np.array(sorted(vocabulary), dtype=np.int32)
        self.cfg = cfg
        self.mu = rng_for(cfg['latent_seed'], 1).uniform(cfg['timing_mean_min_ms'], cfg['timing_mean_max_ms'], len(self.tokens))
        self.patterns = np.zeros((len(self.tokens), cfg['cache_width']), dtype=bool)
        for i, token in enumerate(self.tokens):
            key = int(token) if cfg['cache_fingerprint_groups'] is None else i % cfg['cache_fingerprint_groups']
            rng = np.random.default_rng(key + 12345)
            count = max(1, int(np.clip(rng.normal(12, 3), 3, 30)))
            self.patterns[i, rng.choice(cfg['cache_width'], count, replace=False)] = True

    def observe(self, true_tokens, seed, component):
        index = np.searchsorted(self.tokens, true_tokens)
        if np.any(index >= len(self.tokens)) or not np.array_equal(self.tokens[index], true_tokens):
            raise ValueError('Victim token outside the declared public vocabulary')
        rng = rng_for(seed, component)
        timing = rng.normal(self.mu[index], self.cfg['timing_intrinsic_sigma_ms'])
        if self.cfg['external_noise_ms']:
            timing += rng.normal(0, self.cfg['external_noise_ms'], len(index))
        probability = np.where(self.patterns[index], self.cfg['cache_hit_probability'], self.cfg['cache_false_positive_probability'])
        cache = rng.random(probability.shape) < probability
        return Observations(timing, cache)

    def profile(self, seed, output):
        n = self.cfg['profile_samples_per_type']
        rng = rng_for(seed, 2)
        timings = rng.normal(self.mu[:, None], self.cfg['timing_intrinsic_sigma_ms'], (len(self.tokens), n))
        cache = np.empty(self.patterns.shape, dtype=np.float64)
        packed = np.empty((len(self.tokens), n, (self.cfg['cache_width'] + 7) // 8), dtype=np.uint8)
        for start in range(0, len(self.tokens), 128):
            end = min(start + 128, len(self.tokens))
            probability = np.where(self.patterns[start:end], self.cfg['cache_hit_probability'], self.cfg['cache_false_positive_probability'])
            measured = rng.random((end-start, n, self.cfg['cache_width'])) < probability[:, None, :]
            cache[start:end] = measured.mean(axis=1)
            packed[start:end] = np.packbits(measured, axis=-1)
        profiles = Profiles(self.tokens, timings.mean(axis=1), np.maximum(timings.std(axis=1, ddof=1), 1e-6), cache)
        np.savez_compressed(output / 'profile_measurements.npz', token_ids=self.tokens, timing_ms=timings, cache_packed=packed)
        np.savez_compressed(output / 'profiles.npz', token_ids=self.tokens, mu=profiles.mu, sigma=profiles.sigma, cache=cache)
        return profiles
