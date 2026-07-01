import numpy as np
from scipy import stats
from option import Option
from gbm import MCEngine


def make_engine(**kw):
    S0     = kw.pop('S0', 100)
    mu     = kw.pop('mu', 0.08)
    sigma  = kw.pop('sigma', 0.2)
    T      = kw.pop('T', 1)
    n_steps = kw.pop('n_steps', 252)
    n_paths = kw.pop('n_paths', 50_000)
    seed   = kw.pop('seed', 0)
    opt = Option(S=S0, K=100, T=T, r=mu, sigma=sigma, kind='call')
    return MCEngine(opt, n_steps=n_steps, n_paths=n_paths, seed=seed)


# --- Shape et valeurs de base ---

def test_shape():
    engine = make_engine(n_paths=1_000, n_steps=50)
    assert engine.simulate().shape == (1_000, 51)

def test_starts_at_S0():
    engine = make_engine()
    assert np.all(engine.simulate()[:, 0] == engine.S0)

def test_all_paths_positive():
    assert np.all(make_engine().simulate() > 0)


# --- Reproductibilité ---

def test_same_seed_reproducible():
    paths1 = make_engine(seed=42).simulate()
    paths2 = make_engine(seed=42).simulate()
    np.testing.assert_array_equal(paths1, paths2)

def test_different_seeds_differ():
    paths1 = make_engine(seed=0).simulate()
    paths2 = make_engine(seed=1).simulate()
    assert not np.array_equal(paths1, paths2)


# --- Moments des log-rendements ---

def test_log_returns_mean():
    engine = make_engine()
    log_returns = np.log(engine.simulate()[:, -1] / engine.S0)
    expected = (engine.mu - 0.5 * engine.sigma ** 2) * engine.T
    assert abs(log_returns.mean() - expected) < 0.01

def test_log_returns_variance():
    engine = make_engine()
    log_returns = np.log(engine.simulate()[:, -1] / engine.S0)
    expected = engine.sigma ** 2 * engine.T
    assert abs(log_returns.var() - expected) < 0.01


# --- Distribution log-normale (test KS) ---

def test_log_returns_normality():
    engine = make_engine()
    log_returns = np.log(engine.simulate()[:, -1] / engine.S0)
    mu_th    = (engine.mu - 0.5 * engine.sigma ** 2) * engine.T
    sigma_th = engine.sigma * np.sqrt(engine.T)
    standardised = (log_returns - mu_th) / sigma_th
    _, p_value = stats.kstest(standardised, 'norm')
    assert p_value > 0.05  # on ne rejette pas la normalité
