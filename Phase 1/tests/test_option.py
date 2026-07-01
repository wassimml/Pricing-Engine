import pytest
from pathlib import Path

from option import Option


def test_valid_call():
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    assert opt.kind == 'call'

def test_valid_put():
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put')
    assert opt.kind == 'put'

def test_T_zero_raises():
    with pytest.raises(ValueError):
        Option(S=100, K=100, T=0, r=0.05, sigma=0.2, kind='call')

def test_T_negative_raises():
    with pytest.raises(ValueError):
        Option(S=100, K=100, T=-1, r=0.05, sigma=0.2, kind='call')

def test_sigma_zero_raises():
    with pytest.raises(ValueError):
        Option(S=100, K=100, T=1, r=0.05, sigma=0, kind='call')

def test_sigma_negative_raises():
    with pytest.raises(ValueError):
        Option(S=100, K=100, T=1, r=0.05, sigma=-0.2, kind='call')

def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='forward')
