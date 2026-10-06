"""Regression and sanity tests for PQC and DensityEstimator."""
import gc

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import adjusted_rand_score

from apqc.density_estimation import DensityEstimator
from apqc.pqc import PQC


def two_blobs(seed=0, n=25):
    """Two tight, well-separated 3-D blobs with their generating labels."""
    rng = np.random.default_rng(seed)
    x = np.r_[rng.normal(-1, .15, (n, 3)), rng.normal(1, .15, (n, 3))].astype("float32")
    return x, np.repeat([0, 1], n)


def fit(x, sigma, steps=200, seed=0):
    """Fit PQC with a scalar bandwidth and return the model."""
    tf.keras.utils.set_random_seed(seed)
    model = PQC(x, batch=len(x), force_cpu=True, optimizer=tf.keras.optimizers.Adam(learning_rate=.01))
    model.set_sigmas(sigma_value=sigma)
    model.cluster_allocation_by_sgd(steps=steps, patience=8)
    model.cluster_allocation_by_probability()
    return model


def test_recovers_separated_blobs():
    x, truth = two_blobs()
    model = fit(x, .2)
    assert adjusted_rand_score(truth, model.proba_labels) == pytest.approx(1.0)


def test_step_budget_is_respected():
    x, _ = two_blobs()
    model = fit(x, .2, steps=5)
    assert int(model.step.numpy()) <= 5


def test_repeated_fits_after_prior_models_are_collected():
    # Used to fail with dead weak references to an earlier fit's variable.
    for seed in range(4):
        x = np.random.default_rng(seed).normal(size=(40, 3)).astype("float32")
        model = fit(x, .4, steps=25, seed=seed)
        confidence, _, _ = model.cluster_probability_per_sample_batched(x[:5], model.sgd_labels)
        assert np.isfinite(confidence).all()
        del model
        gc.collect()


def test_probabilities_are_valid_and_models_independent():
    x, _ = two_blobs(910)
    model = fit(x, .2, steps=100)
    query = x[:8] / float(model.scale.numpy())
    confidence, labels, _ = model.cluster_probability_per_sample_batched(query, model.sgd_labels)
    assert np.all((confidence >= 0) & (confidence <= 1 + 1e-6))
    # Reversing the queries reverses the answers.
    rev_conf, rev_labels, _ = model.cluster_probability_per_sample_batched(query[::-1].copy(), model.sgd_labels)
    np.testing.assert_equal(labels, rev_labels[::-1])
    np.testing.assert_allclose(confidence, rev_conf[:, ::-1], atol=1e-5)
    # Fitting another model must not alter the first one.
    fit(x * 1.1, .4, steps=100, seed=911)
    after_conf, after_labels, _ = model.cluster_probability_per_sample_batched(query, model.sgd_labels)
    np.testing.assert_equal(labels, after_labels)
    np.testing.assert_allclose(confidence, after_conf, atol=1e-5)


def test_knn_bandwidth_and_energy_merge():
    x, _ = two_blobs(3)
    tf.keras.utils.set_random_seed(3)
    model = PQC(x, batch=len(x), force_cpu=True)
    model.set_sigmas(knn_ratio=.2)
    model.cluster_allocation_by_sgd(steps=200)
    model.cluster_allocation_by_probability()
    if model.k_num >= 2:
        assert model.hierarchical_energy_merge()
        merged = model.energy_merge_results[model.sigmas_id]["merged_sgd_labels"]
        assert merged.shape[0] == len(x)
        # Last merge level joins every well into a single cluster.
        assert len(np.unique(merged[:, -1])) == 1


def test_density_estimator_default_scale():
    x, _ = two_blobs(5)
    estimator = DensityEstimator(x, batch=len(x), force_cpu=True)
    assert float(estimator.scale.numpy()) > 0
    losses = estimator.fit(steps=20)
    assert np.isfinite(losses).all()
