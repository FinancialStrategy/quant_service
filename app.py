from __future__ import annotations

from typing import Dict, List, Optional, Literal
import time
import numpy as np
import os
import pandas as pd
import cvxpy as cp
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.black_litterman import BlackLittermanModel
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR, EfficientCDaR, EfficientSemivariance
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.cla import CLA

app = FastAPI(title='MK PortfolioOPTIM Service', version='0.13.0')

QUANT_SERVICE_SECRET = os.getenv('QUANT_SERVICE_SECRET', '').strip()


def _authorize(authorization: Optional[str]):
    if not QUANT_SERVICE_SECRET:
        return
    expected = f'Bearer {QUANT_SERVICE_SECRET}'
    if authorization != expected:
        raise HTTPException(401, 'Unauthorized quant-service request.')


class FactorConstraint(BaseModel):
    factor: str
    lower: float
    upper: float
    loadings: Dict[str, Optional[float]]


class GroupConstraint(BaseModel):
    name: str
    members: List[str]
    lower: float = 0.0
    upper: float = 1.0


class RelativeView(BaseModel):
    long: str
    short: str
    view: float
    confidence: float = 0.50


class OptimizeRequest(BaseModel):
    tickers: List[str]
    dates: List[str]
    prices: List[List[Optional[float]]]
    method: Literal[
        'max_sharpe','min_volatility','max_quadratic_utility','efficient_return','efficient_risk',
        'black_litterman','risk_parity','hrp','cla_min_volatility','cla_max_sharpe','min_semivariance','min_cvar','min_cdar'
    ] = 'max_sharpe'
    risk_free_rate: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    risk_aversion: float = 1.0
    l2_gamma: float = 0.0
    cvar_beta: float = 0.95
    cdar_beta: float = 0.95
    factor_constraints: List[FactorConstraint] = Field(default_factory=list)
    group_constraints: List[GroupConstraint] = Field(default_factory=list)
    absolute_views: Dict[str, float] = Field(default_factory=dict)
    relative_views: List[RelativeView] = Field(default_factory=list)
    view_confidences: Dict[str, float] = Field(default_factory=dict)
    bl_tau: float = 0.05
    risk_budgets: Dict[str, float] = Field(default_factory=dict)
    current_weights: Dict[str, float] = Field(default_factory=dict)
    benchmark_weights: Dict[str, float] = Field(default_factory=dict)
    turnover_limit: Optional[float] = None
    turnover_penalty: float = 0.0
    tracking_error_limit: Optional[float] = None
    min_effective_n: Optional[float] = None
    liquidity_caps: Dict[str, float] = Field(default_factory=dict)
    return_model: Literal['historical_mean','ema_mean','capm'] = 'historical_mean'
    risk_model: Literal['ledoit_wolf','sample_cov','ewma_cov'] = 'ledoit_wolf'
    ema_span: int = 500
    ewma_span: int = 180
    frontier_points: int = 81


class WalkForwardRequest(BaseModel):
    request: OptimizeRequest
    train_days: int = 756
    test_days: int = 63
    step_days: int = 63
    max_folds: int = 10
    cost_bps: float = 10.0
    strategies: List[Literal['max_sharpe','min_volatility','risk_parity','black_litterman','equal_weight']] = Field(
        default_factory=lambda: ['max_sharpe','min_volatility','risk_parity','black_litterman','equal_weight']
    )


class RobustnessRequest(BaseModel):
    request: OptimizeRequest
    strategy: Literal['max_sharpe','min_volatility','risk_parity','black_litterman'] = 'max_sharpe'
    return_models: List[Literal['historical_mean','ema_mean','capm']] = Field(default_factory=lambda: ['historical_mean','ema_mean'])
    risk_models: List[Literal['ledoit_wolf','sample_cov','ewma_cov']] = Field(default_factory=lambda: ['ledoit_wolf','sample_cov','ewma_cov'])
    bootstrap_count: int = 20
    bootstrap_seed: int = 2026


def _frame(req: OptimizeRequest) -> pd.DataFrame:
    if len(req.tickers) < 2:
        raise HTTPException(400, 'At least two assets are required.')
    if len(req.dates) != len(req.prices):
        raise HTTPException(400, 'dates/prices length mismatch.')
    df = pd.DataFrame(req.prices, index=pd.to_datetime(req.dates), columns=req.tickers, dtype=float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    if len(df) < 60:
        raise HTTPException(400, 'At least 60 complete price observations are required.')
    if (df <= 0).any().any():
        raise HTTPException(400, 'Prices must be positive.')
    return df


def _clone_request(req: OptimizeRequest, **updates) -> OptimizeRequest:
    raw = req.model_dump() if hasattr(req, 'model_dump') else req.dict()
    raw.update(updates)
    return OptimizeRequest(**raw)


def _model(prices: pd.DataFrame, req: OptimizeRequest):
    returns = expected_returns.returns_from_prices(prices).dropna(how='any')
    if req.return_model == 'ema_mean':
        mu = expected_returns.ema_historical_return(prices, span=max(20, int(req.ema_span)), frequency=252)
    elif req.return_model == 'capm':
        mu = expected_returns.capm_return(prices, risk_free_rate=req.risk_free_rate, frequency=252)
    else:
        mu = expected_returns.mean_historical_return(prices, frequency=252)

    if req.risk_model == 'sample_cov':
        cov = risk_models.sample_cov(prices, frequency=252)
    elif req.risk_model == 'ewma_cov':
        cov = risk_models.exp_cov(prices, span=max(20, int(req.ewma_span)), frequency=252)
    else:
        cov = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
    return returns, mu, cov


def _current_vec(req: OptimizeRequest, tickers: List[str]) -> np.ndarray:
    arr = np.array([float(req.current_weights.get(t, 0.0)) for t in tickers], dtype=float)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    return arr


def _benchmark_vec(req: OptimizeRequest, tickers: List[str]) -> np.ndarray:
    arr = np.array([float(req.benchmark_weights.get(t, 0.0)) for t in tickers], dtype=float)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    return arr


def _weight_bounds(req: OptimizeRequest, tickers: List[str]):
    out = []
    for t in tickers:
        hi = min(float(req.upper_bound), float(req.liquidity_caps.get(t, req.upper_bound)))
        lo = float(req.lower_bound)
        if hi < lo - 1e-12:
            raise HTTPException(400, f'Liquidity cap for {t} is below lower weight bound.')
        out.append((lo, hi))
    return out


def _constraint_precheck(req: OptimizeRequest, tickers: List[str]):
    issues, warnings = [], []
    try:
        bounds = _weight_bounds(req, tickers)
        min_sum = sum(lo for lo, _ in bounds)
        max_sum = sum(hi for _, hi in bounds)
        if min_sum > 1.0 + 1e-9:
            issues.append(f'Sum of lower/liquidity-adjusted bounds is {min_sum:.4f} > 1.')
        if max_sum < 1.0 - 1e-9:
            issues.append(f'Sum of upper/liquidity-adjusted bounds is {max_sum:.4f} < 1.')
    except HTTPException as exc:
        issues.append(str(exc.detail))
    if req.min_effective_n is not None and req.min_effective_n > len(tickers) + 1e-9:
        issues.append(f'Minimum Effective N {req.min_effective_n:g} exceeds asset count {len(tickers)}.')
    for gc in req.group_constraints:
        if gc.lower > gc.upper:
            issues.append(f'Group constraint {gc.name} has lower > upper.')
        members = [t for t in gc.members if t in tickers]
        if not members:
            warnings.append(f'Group constraint {gc.name} has no selected-universe members and is ignored.')
    if req.turnover_limit is not None and req.turnover_limit < 0:
        issues.append('Turnover limit must be non-negative.')
    if req.tracking_error_limit is not None and req.tracking_error_limit <= 0:
        warnings.append('Tracking-error limit is non-positive and is ignored.')
    return {'status': 'FAIL' if issues else 'PASS', 'issues': issues, 'warnings': warnings}


def _apply_constraints(opt, req: OptimizeRequest, tickers: List[str], cov: Optional[pd.DataFrame] = None):
    for fc in req.factor_constraints:
        beta = np.array([fc.loadings.get(t) for t in tickers], dtype=float)
        if np.isnan(beta).any():
            raise HTTPException(400, f'Missing {fc.factor} loading for one or more assets.')
        if fc.lower > fc.upper:
            raise HTTPException(400, f'Invalid bounds for factor {fc.factor}.')
        opt.add_constraint(lambda w, b=beta, lo=float(fc.lower): w @ b >= lo)
        opt.add_constraint(lambda w, b=beta, hi=float(fc.upper): w @ b <= hi)

    for gc in req.group_constraints:
        idx = [i for i, t in enumerate(tickers) if t in set(gc.members)]
        if not idx:
            continue
        lo, hi = float(gc.lower), float(gc.upper)
        opt.add_constraint(lambda w, ix=idx, lo=lo: cp.sum(w[ix]) >= lo)
        opt.add_constraint(lambda w, ix=idx, hi=hi: cp.sum(w[ix]) <= hi)

    if req.min_effective_n is not None and req.min_effective_n > 1:
        cap = 1.0 / float(req.min_effective_n)
        opt.add_constraint(lambda w, cap=cap: cp.sum_squares(w) <= cap)

    cur = _current_vec(req, tickers)
    if req.turnover_limit is not None and req.turnover_limit >= 0 and cur.sum() > 0:
        limit = float(req.turnover_limit)
        opt.add_constraint(lambda w, c=cur, lim=limit: cp.norm1(w - c) <= lim)
    if req.turnover_penalty > 0 and cur.sum() > 0:
        penalty = float(req.turnover_penalty)
        opt.add_objective(lambda w, c=cur, p=penalty: p * cp.norm1(w - c))

    bench = _benchmark_vec(req, tickers)
    if req.tracking_error_limit is not None and req.tracking_error_limit > 0 and bench.sum() > 0 and cov is not None:
        te2 = float(req.tracking_error_limit) ** 2
        sigma = cp.psd_wrap(np.asarray(cov.values, dtype=float))
        opt.add_constraint(lambda w, b=bench, s=sigma, lim=te2: cp.quad_form(w - b, s) <= lim)


def _factor_exposure(weights: Dict[str, float], constraints: List[FactorConstraint], tickers: List[str]):
    if not constraints:
        return {}
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    out = {}
    for fc in constraints:
        b = np.array([fc.loadings[t] for t in tickers], dtype=float)
        out[fc.factor] = float(np.dot(w, b))
    return out


def _scipy_constraints(req: OptimizeRequest, tickers: List[str], cov: Optional[pd.DataFrame] = None):
    cons = [{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}]
    for fc in req.factor_constraints:
        beta = np.array([fc.loadings.get(t) for t in tickers], dtype=float)
        if np.isnan(beta).any():
            raise HTTPException(400, f'Missing {fc.factor} loading for one or more assets.')
        cons.append({'type': 'ineq', 'fun': lambda w, b=beta, lo=float(fc.lower): float(w @ b - lo)})
        cons.append({'type': 'ineq', 'fun': lambda w, b=beta, hi=float(fc.upper): float(hi - w @ b)})
    for gc in req.group_constraints:
        idx = [i for i, t in enumerate(tickers) if t in set(gc.members)]
        if not idx:
            continue
        cons.append({'type': 'ineq', 'fun': lambda w, ix=idx, lo=float(gc.lower): float(np.sum(w[ix]) - lo)})
        cons.append({'type': 'ineq', 'fun': lambda w, ix=idx, hi=float(gc.upper): float(hi - np.sum(w[ix]))})
    if req.min_effective_n is not None and req.min_effective_n > 1:
        cap = 1.0 / float(req.min_effective_n)
        cons.append({'type': 'ineq', 'fun': lambda w, cap=cap: float(cap - np.sum(w * w))})
    cur = _current_vec(req, tickers)
    if req.turnover_limit is not None and req.turnover_limit >= 0 and cur.sum() > 0:
        lim = float(req.turnover_limit)
        cons.append({'type': 'ineq', 'fun': lambda w, c=cur, lim=lim: float(lim - np.sum(np.abs(w - c)))})
    bench = _benchmark_vec(req, tickers)
    if req.tracking_error_limit is not None and req.tracking_error_limit > 0 and bench.sum() > 0 and cov is not None:
        sigma = cov.values
        lim2 = float(req.tracking_error_limit) ** 2
        cons.append({'type': 'ineq', 'fun': lambda w, b=bench, s=sigma, lim2=lim2: float(lim2 - (w - b) @ s @ (w - b))})
    return cons


def _risk_parity_weights(cov: pd.DataFrame, req: OptimizeRequest) -> Dict[str, float]:
    tickers = list(cov.index)
    n = len(tickers)
    raw_budget = np.array([float(req.risk_budgets.get(t, 1.0)) for t in tickers], dtype=float)
    if (raw_budget < 0).any() or raw_budget.sum() <= 0:
        raise HTTPException(400, 'Risk budgets must be non-negative and sum to a positive value.')
    budget = raw_budget / raw_budget.sum()
    sigma = cov.values

    def rc_pct(w):
        port_var = float(w @ sigma @ w)
        if port_var <= 0:
            return np.ones(n) / n
        mrc = sigma @ w / np.sqrt(port_var)
        rc = w * mrc
        total = rc.sum()
        return rc / total if abs(total) > 1e-15 else np.ones(n) / n

    def objective(w):
        d = rc_pct(w) - budget
        obj = float(d @ d)
        cur = _current_vec(req, tickers)
        if req.turnover_penalty > 0 and cur.sum() > 0:
            obj += float(req.turnover_penalty) * float(np.sum(np.abs(w - cur)))
        return obj

    x0 = np.ones(n) / n
    bounds = _weight_bounds(req, tickers)
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=_scipy_constraints(req, tickers, cov),
                   options={'maxiter': 2500, 'ftol': 1e-12})
    if not res.success:
        raise HTTPException(422, f'Risk parity optimization failed: {res.message}')
    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    w = w / w.sum()
    return {t: float(v) for t, v in zip(tickers, w)}


def _performance(weights: Dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float):
    w = np.array([weights.get(k, 0.0) for k in mu.index], dtype=float)
    er = float(np.dot(w, mu.values))
    vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
    sharpe = float((er - rf) / vol) if vol > 0 else None
    return {'expected_return': er, 'volatility': vol, 'sharpe': sharpe}


def _raw_optimizer_weights(opt, tickers: List[str]) -> Dict[str, float]:
    raw = getattr(opt, 'weights', None)
    if raw is None:
        cleaned = opt.clean_weights()
        return {t: float(cleaned.get(t, 0.0)) for t in tickers}
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if len(arr) != len(tickers):
        cleaned = opt.clean_weights()
        return {t: float(cleaned.get(t, 0.0)) for t in tickers}
    arr[np.abs(arr) < 1e-12] = 0.0
    total = float(arr.sum())
    if abs(total) > 1e-12:
        arr = arr / total
    return {t: float(v) for t, v in zip(tickers, arr)}


def _risk_decomposition(weights: Dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float):
    tickers = list(mu.index)
    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    vol = float(np.sqrt(max(w @ cov.values @ w, 0.0)))
    if vol > 0:
        mrc = cov.values @ w / vol
        rc = w * mrc
        rc_pct = rc / rc.sum() if abs(rc.sum()) > 1e-15 else np.zeros_like(rc)
    else:
        mrc = np.zeros_like(w); rc = np.zeros_like(w); rc_pct = np.zeros_like(w)
    hhi = float(np.sum(w * w))
    effective_n = float(1.0 / hhi) if hhi > 0 else None
    asset_vols = np.sqrt(np.maximum(np.diag(cov.values), 0.0))
    weighted_asset_vol = float(np.dot(np.abs(w), asset_vols))
    diversification_ratio = float(weighted_asset_vol / vol) if vol > 0 else None
    corr = cov.values / np.outer(asset_vols, asset_vols)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    clusters = {t: 1 for t in tickers}
    if len(tickers) >= 3:
        try:
            dist = np.clip(1.0 - corr, 0.0, 2.0)
            np.fill_diagonal(dist, 0.0)
            z = linkage(squareform(dist, checks=False), method='average')
            k = min(4, max(2, int(round(np.sqrt(len(tickers))))))
            labels = fcluster(z, t=k, criterion='maxclust')
            clusters = {t: int(c) for t, c in zip(tickers, labels)}
        except Exception:
            pass
    return {
        'marginal_risk_contribution': {t: float(v) for t, v in zip(tickers, mrc)},
        'risk_contribution': {t: float(v) for t, v in zip(tickers, rc)},
        'risk_contribution_pct': {t: float(v) for t, v in zip(tickers, rc_pct)},
        'hhi': hhi,
        'effective_n': effective_n,
        'diversification_ratio': diversification_ratio,
        'max_weight': float(np.max(w)) if len(w) else None,
        'correlation_clusters': clusters,
        'performance': _performance(weights, mu, cov, rf),
    }


def _benchmark_record(weights: Dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float,
                      optimization_performance: Optional[Dict[str, float]] = None, **extra):
    common = _performance(weights, mu, cov, rf)
    out = {
        'weights': weights,
        'performance': common,
        'common_performance': common,
        'risk_decomposition': _risk_decomposition(weights, mu, cov, rf),
    }
    if optimization_performance is not None:
        out['optimization_performance'] = optimization_performance
    out.update(extra)
    return out


def _bl_inputs(mu: pd.Series, cov: pd.DataFrame, req: OptimizeRequest):
    tickers = list(mu.index)
    rows = []
    q = []
    conf = []
    labels = []
    for asset, view in req.absolute_views.items():
        if asset not in tickers:
            raise ValueError(f'Black-Litterman view asset {asset} is not in selected universe.')
        row = np.zeros(len(tickers)); row[tickers.index(asset)] = 1.0
        rows.append(row); q.append(float(view))
        conf.append(float(req.view_confidences.get(asset, 0.50)))
        labels.append(asset)
    for rv in req.relative_views:
        if rv.long not in tickers or rv.short not in tickers:
            raise ValueError(f'Black-Litterman relative view {rv.long}>{rv.short} uses asset outside selected universe.')
        row = np.zeros(len(tickers)); row[tickers.index(rv.long)] = 1.0; row[tickers.index(rv.short)] = -1.0
        rows.append(row); q.append(float(rv.view)); conf.append(float(rv.confidence)); labels.append(f'{rv.long}>{rv.short}')
    if not rows:
        raise ValueError('No Black-Litterman absolute or relative views supplied.')
    P = np.vstack(rows); Q = np.asarray(q, dtype=float)
    tau = max(1e-6, min(float(req.bl_tau), 1.0))
    base = tau * (P @ cov.values @ P.T)
    omega = np.zeros((len(rows), len(rows)), dtype=float)
    for i, c in enumerate(conf):
        c = min(max(float(c), 1e-4), 0.9999)
        omega[i, i] = max(float(base[i, i]) * (1.0 - c) / c, 1e-10)
    return P, Q, omega, labels


def _black_litterman_posterior(mu: pd.Series, cov: pd.DataFrame, req: OptimizeRequest):
    P, Q, omega, labels = _bl_inputs(mu, cov, req)
    bl = BlackLittermanModel(cov, pi=mu, P=P, Q=Q, omega=omega, tau=max(1e-6, min(float(req.bl_tau), 1.0)))
    return bl.bl_returns(), bl.bl_cov(), labels


def _solve_strategy(mu: pd.Series, cov: pd.DataFrame, returns: pd.DataFrame, req: OptimizeRequest, method: str,
                    apply_l2: bool = True):
    tickers = list(mu.index)
    bounds = _weight_bounds(req, tickers)
    if method in {'max_sharpe','min_volatility','max_quadratic_utility','efficient_return','efficient_risk'}:
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        _apply_constraints(ef, req, tickers, cov)
        if apply_l2 and req.l2_gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
        if method == 'max_sharpe':
            ef.max_sharpe(risk_free_rate=req.risk_free_rate)
        elif method == 'min_volatility':
            ef.min_volatility()
        elif method == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=req.risk_aversion)
        elif method == 'efficient_return':
            if req.target_return is None:
                raise HTTPException(400, 'target_return is required.')
            ef.efficient_return(req.target_return)
        else:
            if req.target_volatility is None:
                raise HTTPException(400, 'target_volatility is required.')
            ef.efficient_risk(req.target_volatility)
        weights = _raw_optimizer_weights(ef, tickers)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {}

    if method == 'black_litterman':
        bl_mu, bl_cov, labels = _black_litterman_posterior(mu, cov, req)
        ef = EfficientFrontier(bl_mu, bl_cov, weight_bounds=bounds)
        _apply_constraints(ef, req, tickers, bl_cov)
        if apply_l2 and req.l2_gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
        ef.max_sharpe(risk_free_rate=req.risk_free_rate)
        weights = _raw_optimizer_weights(ef, tickers)
        posterior = _performance(weights, bl_mu, bl_cov, req.risk_free_rate)
        extra = {
            'prior': req.return_model,
            'posterior_returns': {k: float(v) for k, v in bl_mu.items()},
            'posterior_performance': posterior,
            'view_labels': labels,
        }
        return weights, posterior, extra

    if method == 'risk_parity':
        weights = _risk_parity_weights(cov, req)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {'risk_budget': req.risk_budgets}

    if method == 'hrp':
        unsupported = bool(
            req.factor_constraints or req.group_constraints or req.liquidity_caps
            or req.turnover_limit is not None or req.tracking_error_limit is not None
            or req.min_effective_n is not None or float(req.turnover_penalty or 0) > 0
        )
        if unsupported:
            raise HTTPException(400, 'Advanced linear/quadratic constraints are not supported for HRP. Clear factor/group/liquidity/turnover/tracking-error/effective-N controls or use an EfficientFrontier strategy.')
        opt = HRPOpt(returns)
        weights = {k: float(v) for k, v in opt.optimize().items()}
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {}

    if method in {'cla_min_volatility','cla_max_sharpe'}:
        unsupported = bool(
            req.factor_constraints or req.group_constraints or req.liquidity_caps
            or req.turnover_limit is not None or req.tracking_error_limit is not None
            or req.min_effective_n is not None or float(req.turnover_penalty or 0) > 0
        )
        if unsupported:
            raise HTTPException(400, 'Advanced constraints are not supported for CLA. Clear factor/group/liquidity/turnover/tracking-error/effective-N controls or use an EfficientFrontier strategy.')
        opt = CLA(mu, cov, weight_bounds=bounds)
        if method == 'cla_min_volatility': opt.min_volatility()
        else: opt.max_sharpe()
        weights = _raw_optimizer_weights(opt, tickers)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {}

    if method == 'min_semivariance':
        opt = EfficientSemivariance(mu, returns, weight_bounds=bounds)
        _apply_constraints(opt, req, tickers, cov)
        opt.min_semivariance()
        weights = _raw_optimizer_weights(opt, tickers)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {'risk_measure': 'semivariance'}

    if method == 'min_cvar':
        opt = EfficientCVaR(mu, returns, beta=req.cvar_beta, weight_bounds=bounds)
        _apply_constraints(opt, req, tickers, cov)
        opt.min_cvar()
        weights = _raw_optimizer_weights(opt, tickers)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {'beta': req.cvar_beta, 'risk_measure': 'CVaR'}

    if method == 'min_cdar':
        opt = EfficientCDaR(mu, returns, beta=req.cdar_beta, weight_bounds=bounds)
        _apply_constraints(opt, req, tickers, cov)
        opt.min_cdar()
        weights = _raw_optimizer_weights(opt, tickers)
        return weights, _performance(weights, mu, cov, req.risk_free_rate), {'beta': req.cdar_beta, 'risk_measure': 'CDaR'}

    raise HTTPException(400, 'Unsupported optimization method.')


def _exact_min_vol(mu, cov, returns, req):
    w, _, _ = _solve_strategy(mu, cov, returns, req, 'min_volatility', apply_l2=False)
    return _benchmark_record(w, mu, cov, req.risk_free_rate)


def _exact_max_sharpe(mu, cov, returns, req):
    w, _, _ = _solve_strategy(mu, cov, returns, req, 'max_sharpe', apply_l2=False)
    return _benchmark_record(w, mu, cov, req.risk_free_rate)


def _black_litterman_benchmark(mu, cov, returns, req):
    w, post_perf, extra = _solve_strategy(mu, cov, returns, req, 'black_litterman', apply_l2=False)
    return _benchmark_record(
        w, mu, cov, req.risk_free_rate,
        optimization_performance=post_perf,
        comparison_note='Optimized on Black-Litterman posterior; scored on common base model.',
        **extra,
    )


def _benchmarks(mu, cov, returns, req):
    out = {}
    for key, fn in (
        ('min_volatility', lambda: _exact_min_vol(mu, cov, returns, req)),
        ('max_sharpe', lambda: _exact_max_sharpe(mu, cov, returns, req)),
    ):
        try: out[key] = fn()
        except Exception as exc: out[key + '_error'] = str(exc)
    try:
        rp = _risk_parity_weights(cov, req)
        out['risk_parity'] = _benchmark_record(rp, mu, cov, req.risk_free_rate)
    except Exception as exc:
        out['risk_parity_error'] = str(exc)
    try:
        out['black_litterman'] = _black_litterman_benchmark(mu, cov, returns, req)
    except Exception as exc:
        out['black_litterman_error'] = str(exc)
    for key in ('min_volatility','max_sharpe','risk_parity','black_litterman'):
        rec = out.get(key)
        if isinstance(rec, dict) and rec.get('weights'):
            rec['constraint_diagnostics'] = _constraint_diagnostics(rec['weights'], req, list(mu.index), cov)
    return out


def _constraint_diagnostics(weights, req: OptimizeRequest, tickers: List[str], cov: Optional[pd.DataFrame] = None):
    tol = 2e-4
    lower, upper = float(req.lower_bound), float(req.upper_bound)
    binding_lower, binding_upper = [], []
    for t in tickers:
        w = float(weights.get(t, 0.0))
        hi = min(upper, float(req.liquidity_caps.get(t, upper)))
        if abs(w - lower) <= tol: binding_lower.append(t)
        if abs(w - hi) <= tol: binding_upper.append(t)
    factor_rows = []
    for fc in req.factor_constraints:
        vals = np.array([float(fc.loadings[t]) for t in tickers], dtype=float)
        wv = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
        exposure = float(wv @ vals)
        binding = abs(exposure - float(fc.lower)) <= tol or abs(exposure - float(fc.upper)) <= tol
        factor_rows.append({'factor': fc.factor, 'exposure': exposure, 'lower': float(fc.lower), 'upper': float(fc.upper), 'binding': bool(binding)})
    group_rows = []
    for gc in req.group_constraints:
        exposure = float(sum(weights.get(t, 0.0) for t in gc.members))
        group_rows.append({'name': gc.name, 'exposure': exposure, 'lower': float(gc.lower), 'upper': float(gc.upper),
                           'binding': abs(exposure-float(gc.lower)) <= tol or abs(exposure-float(gc.upper)) <= tol})
    cur = _current_vec(req, tickers); wv = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    turnover = float(np.sum(np.abs(wv-cur))) if cur.sum() > 0 else None
    hhi = float(np.sum(wv*wv)); effective_n = float(1/hhi) if hhi > 0 else None
    bench = _benchmark_vec(req, tickers)
    tracking_error = None
    if cov is not None and bench.sum() > 0:
        d = wv-bench; tracking_error = float(np.sqrt(max(d @ cov.values @ d, 0.0)))
    return {
        'solver_status': 'optimal', 'lower_bound': lower, 'upper_bound': upper,
        'binding_lower': binding_lower, 'binding_upper': binding_upper,
        'factor_constraints': factor_rows, 'group_constraints': group_rows,
        'turnover': turnover, 'turnover_limit': req.turnover_limit,
        'effective_n': effective_n, 'min_effective_n': req.min_effective_n,
        'tracking_error': tracking_error, 'tracking_error_limit': req.tracking_error_limit,
        'liquidity_caps': req.liquidity_caps,
    }


def _max_return_feasible(mu, cov, req):
    tickers = list(mu.index); n = len(tickers)
    x0 = np.ones(n) / n
    bnds = _weight_bounds(req, tickers)
    res = minimize(lambda w: -float(w @ mu.values), x0, method='SLSQP', bounds=bnds,
                   constraints=_scipy_constraints(req, tickers, cov), options={'maxiter':2500,'ftol':1e-11})
    if not res.success:
        return float(mu.max())
    return float(res.x @ mu.values)


def _min_vol_point(mu, cov, returns, req):
    w, _, _ = _solve_strategy(mu, cov, returns, req, 'min_volatility', apply_l2=False)
    p = _performance(w, mu, cov, req.risk_free_rate)
    return {'return': p['expected_return'], 'volatility': p['volatility'], 'sharpe': p['sharpe']}


def _frontier(mu, cov, returns, req, n_points: Optional[int] = None):
    n_points = int(n_points or req.frontier_points or 81)
    n_points = max(9, min(n_points, 121))
    try:
        minpt = _min_vol_point(mu, cov, returns, req)
        lo = float(minpt['return']); hi = _max_return_feasible(mu, cov, req)
    except Exception:
        return []
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        return [minpt]
    pts = []
    for target in np.linspace(lo, lo + (hi-lo)*0.999, n_points):
        try:
            tmp = _clone_request(req, target_return=float(target), l2_gamma=0.0)
            w, _, _ = _solve_strategy(mu, cov, returns, tmp, 'efficient_return', apply_l2=False)
            p = _performance(w, mu, cov, req.risk_free_rate)
            pts.append({'return':p['expected_return'],'volatility':p['volatility'],'sharpe':p['sharpe']})
        except Exception:
            continue
    try:
        tan = _exact_max_sharpe(mu, cov, returns, req)['performance']
        pts.append({'return':float(tan['expected_return']),'volatility':float(tan['volatility']),'sharpe':float(tan['sharpe'])})
    except Exception:
        pass
    out = []
    for p in sorted(pts, key=lambda x:x['volatility']):
        if not out or abs(p['volatility']-out[-1]['volatility'])>1e-7 or abs(p['return']-out[-1]['return'])>1e-7:
            out.append(p)
    return out


def _daily_metrics(rets: np.ndarray, rf: float):
    x = np.asarray(rets, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    nav = np.cumprod(1.0 + x)
    years = len(x) / 252.0
    cagr = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 and nav[-1] > 0 else None
    vol = float(np.std(x, ddof=1) * np.sqrt(252)) if len(x) > 1 else 0.0
    excess = x - rf / 252.0
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)) if len(x)>1 and np.std(excess, ddof=1)>0 else None
    down = excess[excess < 0]
    sortino = float(np.mean(excess) / np.std(down, ddof=1) * np.sqrt(252)) if len(down)>1 and np.std(down, ddof=1)>0 else None
    peak = np.maximum.accumulate(nav); dd = nav / peak - 1.0
    mdd = float(np.min(dd))
    q = float(np.quantile(x, 0.01)); tail = x[x <= q]
    return {
        'cagr': cagr, 'volatility': vol, 'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': mdd, 'calmar': float(cagr/abs(mdd)) if cagr is not None and mdd < 0 else None,
        'var99_1d': q, 'cvar99_1d': float(np.mean(tail)) if len(tail) else q,
        'positive_days': float(np.mean(x > 0)), 'terminal_nav': float(nav[-1]),
    }


def _solve_for_fold(train_prices: pd.DataFrame, req: OptimizeRequest, method: str):
    returns, mu, cov = _model(train_prices, req)
    if method == 'equal_weight':
        w = {t: 1.0/len(mu) for t in mu.index}
        return w, _performance(w, mu, cov, req.risk_free_rate)
    w, perf, _ = _solve_strategy(mu, cov, returns, req, method, apply_l2=False)
    return w, perf


def _bootstrap_prices(prices: pd.DataFrame, rng: np.random.Generator):
    rets = prices.pct_change().dropna(how='any')
    idx = rng.integers(0, len(rets), size=len(rets))
    sampled = rets.iloc[idx].reset_index(drop=True)
    arr = np.vstack([np.ones(prices.shape[1]), np.cumprod(1.0 + sampled.values, axis=0)])
    return pd.DataFrame(arr * 100.0, columns=prices.columns)


@app.get('/health')
def health():
    return {'ok': True, 'engine': 'PortfolioOPTIM', 'version': '0.13.0', 'auth_required': bool(QUANT_SERVICE_SECRET)}


@app.post('/optimize')
def optimize(req: OptimizeRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    t0 = time.perf_counter()
    prices = _frame(req)
    precheck = _constraint_precheck(req, list(prices.columns))
    if precheck['issues']:
        raise HTTPException(400, 'Constraint precheck failed: ' + ' | '.join(precheck['issues']))
    returns, mu, cov = _model(prices, req)
    try:
        weights, perf, extra = _solve_strategy(mu, cov, returns, req, req.method, apply_l2=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'Optimization failed: {exc}')

    t_bench = time.perf_counter()
    benchmarks = _benchmarks(mu, cov, returns, req)
    bench_ms = (time.perf_counter() - t_bench) * 1000.0
    t_front = time.perf_counter()
    frontier = _frontier(mu, cov, returns, req)
    frontier_ms = (time.perf_counter() - t_front) * 1000.0

    result = {
        'engine': 'PortfolioOPTIM', 'version': '0.13.0', 'method': req.method,
        'observations': int(len(prices)), 'data_start': str(prices.index.min().date()), 'data_end': str(prices.index.max().date()),
        'weights': weights, 'performance': perf, 'selected_extra': extra,
        'expected_returns': {k: float(v) for k,v in mu.items()},
        'covariance': {r: {c: float(cov.loc[r,c]) for c in cov.columns} for r in cov.index},
        'correlation': {r: {c: float(returns.corr().loc[r,c]) for c in returns.columns} for r in returns.columns},
        'model_spec': {
            'expected_return_model': req.return_model, 'risk_model': req.risk_model,
            'ema_span': req.ema_span, 'ewma_span': req.ewma_span, 'frequency': 252,
            'comparison_benchmarks_l2_gamma': 0.0, 'comparison_evaluator': 'backend_common_evaluator_v0.13.0',
        },
        'frontier': frontier,
        'factor_exposure': _factor_exposure(weights, req.factor_constraints, list(mu.index)),
        'benchmarks': benchmarks,
        'constraint_diagnostics': _constraint_diagnostics(weights, req, list(mu.index), cov),
        'risk_decomposition': _risk_decomposition(weights, mu, cov, req.risk_free_rate),
        'constraint_precheck': precheck,
        'timing': {
            'benchmarks_ms': round(bench_ms, 1), 'frontier_ms': round(frontier_ms, 1),
            'total_ms': round((time.perf_counter()-t0)*1000.0, 1), 'cache_hit': False,
        }
    }
    return result


@app.post('/walkforward')
def walkforward(req: WalkForwardRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    base = req.request
    prices = _frame(base)
    train_days = max(252, int(req.train_days)); test_days = max(21, int(req.test_days)); step_days = max(test_days, int(req.step_days))
    if len(prices) < train_days + test_days:
        raise HTTPException(400, 'Not enough observations for requested walk-forward windows.')
    fold_ends = list(range(train_days, len(prices)-test_days+1, step_days))
    fold_ends = fold_ends[-max(1, min(int(req.max_folds), 20)):]
    strategies = list(dict.fromkeys(req.strategies))
    series = {s: [] for s in strategies}; dates = []; folds = []
    prev = {s: _current_vec(base, list(prices.columns)) for s in strategies}
    total_turnover = {s: 0.0 for s in strategies}
    cost_rate = max(0.0, float(req.cost_bps)) / 10000.0

    for fold_no, end in enumerate(fold_ends, start=1):
        train = prices.iloc[end-train_days:end].copy(); test = prices.iloc[end:end+test_days].copy()
        test_rets = test.pct_change().dropna(how='any')
        if len(test_rets) == 0: continue
        fold_rec = {'fold': fold_no, 'train_from': str(train.index[0].date()), 'train_to': str(train.index[-1].date()),
                    'test_from': str(test_rets.index[0].date()), 'test_to': str(test_rets.index[-1].date()), 'strategies': {}}
        for s in strategies:
            try:
                local_req = _clone_request(base, method='max_sharpe' if s=='equal_weight' else s, l2_gamma=0.0)
                w_dict, train_perf = _solve_for_fold(train, local_req, s)
                w = np.array([w_dict.get(t, 0.0) for t in prices.columns], dtype=float)
                turnover = float(np.sum(np.abs(w - prev[s]))) if prev[s].sum() > 0 else float(np.sum(np.abs(w)))
                total_turnover[s] += turnover
                r = test_rets.values @ w
                if len(r): r[0] -= turnover * cost_rate
                series[s].extend([float(x) for x in r])
                prev[s] = w
                fold_rec['strategies'][s] = {'weights': w_dict, 'train_performance': train_perf, 'turnover': turnover,
                                             'test_return': float(np.prod(1.0+r)-1.0)}
            except Exception as exc:
                fold_rec['strategies'][s] = {'error': str(exc)}
        dates.extend([str(x.date()) for x in test_rets.index])
        folds.append(fold_rec)

    metrics = {}
    for s, vals in series.items():
        m = _daily_metrics(np.array(vals), base.risk_free_rate)
        years = max(len(vals)/252.0, 1e-9)
        m['annualized_turnover'] = float(total_turnover[s]/years)
        metrics[s] = m
    return {'engine':'PortfolioOPTIM','version':'0.13.0','dates':dates,'returns':series,'metrics':metrics,'folds':folds,
            'settings':{'train_days':train_days,'test_days':test_days,'step_days':step_days,'cost_bps':req.cost_bps,'non_overlapping_test_windows':True}}


@app.post('/robustness')
def robustness(req: RobustnessRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    base = req.request
    prices = _frame(base)
    rows = []
    weight_matrix = []
    tickers = list(prices.columns)
    for rm in req.return_models[:3]:
        for sm in req.risk_models[:3]:
            try:
                local = _clone_request(base, return_model=rm, risk_model=sm, method=req.strategy, l2_gamma=0.0)
                rets, mu, cov = _model(prices, local)
                w, _, _ = _solve_strategy(mu, cov, rets, local, req.strategy, apply_l2=False)
                perf = _performance(w, mu, cov, local.risk_free_rate)
                rows.append({'return_model':rm,'risk_model':sm,'weights':w,'performance':perf})
                weight_matrix.append([w.get(t,0.0) for t in tickers])
            except Exception as exc:
                rows.append({'return_model':rm,'risk_model':sm,'error':str(exc)})

    rng = np.random.default_rng(int(req.bootstrap_seed))
    boot = []
    count = max(0, min(int(req.bootstrap_count), 40))
    for i in range(count):
        try:
            bp = _bootstrap_prices(prices, rng)
            local = _clone_request(base, method=req.strategy, l2_gamma=0.0)
            rets, mu, cov = _model(bp, local)
            w, _, _ = _solve_strategy(mu, cov, rets, local, req.strategy, apply_l2=False)
            boot.append([w.get(t,0.0) for t in tickers])
        except Exception:
            continue
    # Lightweight resampled efficient-frontier envelope.  To stay responsive on
    # free infrastructure we use at most six bootstrap histories and nine points
    # per frontier.  Each frontier still uses exact constrained solves.
    frontier_samples = []
    fr_rng = np.random.default_rng(int(req.bootstrap_seed) + 17)
    for _ in range(min(6, count)):
        try:
            bp = _bootstrap_prices(prices, fr_rng)
            local = _clone_request(base, l2_gamma=0.0, frontier_points=9)
            rets, mu, cov = _model(bp, local)
            fr = _frontier(mu, cov, rets, local, n_points=9)
            if len(fr) >= 5:
                frontier_samples.append(fr)
        except Exception:
            continue
    resampled_frontier = []
    if frontier_samples:
        m = min(len(x) for x in frontier_samples)
        for j in range(m):
            vols = np.array([x[j]['volatility'] for x in frontier_samples], dtype=float)
            rets_ = np.array([x[j]['return'] for x in frontier_samples], dtype=float)
            resampled_frontier.append({
                'q': float(j / max(m-1, 1)),
                'mean_volatility': float(np.mean(vols)), 'p10_volatility': float(np.quantile(vols, 0.10)), 'p90_volatility': float(np.quantile(vols, 0.90)),
                'mean_return': float(np.mean(rets_)), 'p10_return': float(np.quantile(rets_, 0.10)), 'p90_return': float(np.quantile(rets_, 0.90)),
            })

    combined = np.array(weight_matrix, dtype=float) if weight_matrix else np.empty((0,len(tickers)))
    boot_arr = np.array(boot, dtype=float) if boot else np.empty((0,len(tickers)))
    all_arr = np.vstack([x for x in (combined, boot_arr) if len(x)]) if len(combined) or len(boot_arr) else np.empty((0,len(tickers)))
    summary = {}
    if len(all_arr):
        mean_w = np.mean(all_arr, axis=0); std_w = np.std(all_arr, axis=0, ddof=1) if len(all_arr)>1 else np.zeros(len(tickers))
        summary = {
            'mean_weights': {t:float(v) for t,v in zip(tickers,mean_w)},
            'weight_std': {t:float(v) for t,v in zip(tickers,std_w)},
            'p10_weights': {t:float(v) for t,v in zip(tickers,np.quantile(all_arr,0.10,axis=0))},
            'p90_weights': {t:float(v) for t,v in zip(tickers,np.quantile(all_arr,0.90,axis=0))},
            'mean_l1_distance': float(np.mean(np.sum(np.abs(all_arr-mean_w),axis=1))),
            'stability_score': float(max(0.0,1.0-0.5*np.mean(np.sum(np.abs(all_arr-mean_w),axis=1))))
        }
        base_rets, base_mu, base_cov = _model(prices, base)
        mean_dict = summary['mean_weights']
        summary['common_performance'] = _performance(mean_dict, base_mu, base_cov, base.risk_free_rate)
    return {'engine':'PortfolioOPTIM','version':'0.13.0','strategy':req.strategy,'model_grid':rows,
            'bootstrap_successes':len(boot),'summary':summary,'tickers':tickers,'resampled_frontier':resampled_frontier}
