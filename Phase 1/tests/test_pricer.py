import numpy as np
import pytest
from option import Option
from pricer import BSModel

model = BSModel()


def call(**kw):
    defaults = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    return Option(**{**defaults, **kw})

def put(**kw):
    defaults = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put')
    return Option(**{**defaults, **kw})


# --- Parité put-call ---

def test_put_call_parity():
    C = model.price(call())
    P = model.price(put())
    S, K, r, T = 100, 100, 0.05, 1
    assert abs((C - P) - (S - K * np.exp(-r * T))) < 1e-8


# --- Cas limites en S ---

def test_call_S_near_zero():
    # S→0 : call→0
    assert model.price(call(S=1e-6)) < 1e-4

def test_put_S_near_zero():
    # S→0 : put→K·e^{-rT}
    opt = put(S=1e-6)
    assert abs(model.price(opt) - opt.K * np.exp(-opt.r * opt.T)) < 1e-4

def test_call_S_large():
    # S→∞ : call→S - K·e^{-rT}
    opt = call(S=1e6)
    expected = opt.S - opt.K * np.exp(-opt.r * opt.T)
    assert abs(model.price(opt) - expected) < 1.0

def test_put_S_large():
    # S→∞ : put→0
    assert model.price(put(S=1e6)) < 1e-4


# --- Cas limite T→0 ---

def test_call_T_near_zero_itm():
    # ITM, T→0 : call→S-K
    opt = call(S=110, K=100, T=1e-6)
    assert abs(model.price(opt) - 10.0) < 0.01

def test_call_T_near_zero_otm():
    # OTM, T→0 : call→0
    assert model.price(call(S=90, K=100, T=1e-6)) < 1e-4


# --- Cas limite σ→0 ---

def test_call_sigma_near_zero_itm():
    # ITM, σ→0 : call→max(S - K·e^{-rT}, 0)
    opt = call(S=110, K=100, sigma=1e-4)
    expected = max(opt.S - opt.K * np.exp(-opt.r * opt.T), 0)
    assert abs(model.price(opt) - expected) < 0.01

def test_call_sigma_near_zero_otm():
    # OTM, σ→0 : call→0
    assert model.price(call(S=90, K=100, sigma=1e-4)) < 1e-4


# --- Cas K=0 ---

def test_call_K_near_zero():
    # K→0 : call→S (pas de strike = détenir l'action)
    opt = call(K=1e-4)
    assert abs(model.price(opt) - opt.S) < 0.01


# --- Cas r=0, σ→0 ---

def test_call_r0_sigma0_itm():
    # r=0, σ→0 : call→max(S-K, 0)
    opt = call(S=110, K=100, r=0.0, sigma=1e-4)
    assert abs(model.price(opt) - 10.0) < 0.01

def test_call_r0_sigma0_otm():
    opt = call(S=90, K=100, r=0.0, sigma=1e-4)
    assert model.price(opt) < 1e-4


# --- price_grid cohérent avec price scalaire ---

def test_price_grid_matches_scalar():
    spots = np.array([80.0, 100.0, 120.0])
    opt = call()
    grid = model.price_grid(spots, opt)
    scalars = np.array([model.price(call(S=s)) for s in spots])
    np.testing.assert_allclose(grid, scalars, rtol=1e-10)
