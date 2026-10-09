import numpy as np
import pytest
from ser.probe import bootstrap_ci, outer_folds, per_actor_uar, permute_within, run, scores, speaker_norm


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(1, 7), 16)
    y = np.tile(np.repeat(np.arange(8), 2), 6)
    X = rng.normal(size=(len(y), 5)) + 1.5 * y[:, None]
    return X, y, g


def test_loso_folds_hold_out_exactly_one_actor(toy):
    _, y, g = toy
    folds = outer_folds(y, g, "loso")
    assert len(folds) == len(np.unique(g))
    for tr, te in folds:
        assert len(np.unique(g[te])) == 1
        assert not np.isin(g[te], g[tr]).any()


def test_permutation_keeps_the_class_counts_of_every_actor(toy):
    _, y, g = toy
    yp = permute_within(y, g, seed=1)
    assert not np.array_equal(yp, y)
    for a in np.unique(g):
        assert np.array_equal(np.bincount(yp[g == a], minlength=8), np.bincount(y[g == a], minlength=8))


def test_speaker_norm_standardises_every_actor(toy):
    X, _, g = toy
    Z = speaker_norm(X, g)
    for a in np.unique(g):
        assert np.allclose(Z[g == a].mean(0), 0, atol=1e-9)
        assert np.allclose(Z[g == a].std(0), 1, atol=1e-6)


def test_scores_on_perfect_predictions(toy):
    _, y, g = toy
    proba = np.eye(8)[y]
    assert scores(y, proba) == {"uar": 1.0, "acc": 1.0, "f1": 1.0}
    assert np.all(per_actor_uar(y, proba, g) == 1.0)


def test_bootstrap_interval_contains_the_point_estimate(toy):
    _, y, g = toy
    pred = np.where(np.random.default_rng(1).random(len(y)) < 0.7, y, (y + 1) % 8)
    proba = np.eye(8)[pred]
    lo, hi = bootstrap_ci(y, proba, g, reps=2000)
    assert lo <= scores(y, proba)["uar"] <= hi


def test_nested_probe_selects_inside_training_actors_and_learns(toy):
    X, y, g = toy
    r = run(X, y, g, "lr", jobs=1)
    assert r["proba"].shape == (len(y), 8)
    assert np.allclose(r["proba"].sum(1), 1)
    assert r["inner"].shape == (6, 1, 7)
    assert scores(y, r["proba"])["uar"] > 0.5


def test_nested_probe_picks_a_layer_per_fold(toy):
    X, y, g = toy
    rng = np.random.default_rng(2)
    stack = np.stack([rng.normal(size=X.shape), X, rng.normal(size=X.shape)], axis=1)
    r = run(stack, y, g, "lr", layers=[0, 1, 2], jobs=1)
    assert [layer for layer, _ in r["choice"]] == [1] * 6
