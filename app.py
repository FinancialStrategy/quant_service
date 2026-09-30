from __future__ import annotations
from typing import Dict, List, Optional, Literal
import numpy as np
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.black_litterman import BlackLittermanModel
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR, EfficientCDaR, EfficientSemivariance
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.cla import CLA

app = FastAPI(title='MK PyPortfolioOpt Service', version='0.10.6')

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
    absolute_views: Dict[str, float] = Field(default_factory=dict)
    risk_budgets: Dict[str, float] = Field(default_factory=dict)


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


def _apply_factor_constraints(opt, req: OptimizeRequest, tickers: List[str]):
    for fc in req.factor_constraints:
        beta = np.array([fc.loadings.get(t) for t in tickers], dtype=float)
        if np.isnan(beta).any():
            raise HTTPException(400, f'Missing {fc.factor} loading for one or more assets.')
        if fc.lower > fc.upper:
            raise HTTPException(400, f'Invalid bounds for factor {fc.factor}.')
        opt.add_constraint(lambda w, b=beta, lo=float(fc.lower): w @ b >= lo)
        opt.add_constraint(lambda w, b=beta, hi=float(fc.upper): w @ b <= hi)

def _factor_exposure(weights: Dict[str, float], constraints: List[FactorConstraint], tickers: List[str]):
    if not constraints:
        return {}
    w = np.array([weights[t] for t in tickers], dtype=float)
    out={}
    for fc in constraints:
        b=np.array([fc.loadings[t] for t in tickers],dtype=float)
        out[fc.factor]=float(np.dot(w,b))
    return out




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
        if port_var <= 0: return np.ones(n) / n
        mrc = sigma @ w / np.sqrt(port_var)
        rc = w * mrc
        total = rc.sum()
        return rc / total if abs(total) > 1e-15 else np.ones(n) / n

    def objective(w):
        d = rc_pct(w) - budget
        return float(d @ d)

    constraints=[{'type':'eq','fun':lambda w: float(np.sum(w)-1.0)}]
    for fc in req.factor_constraints:
        beta=np.array([fc.loadings.get(t) for t in tickers],dtype=float)
        if np.isnan(beta).any():
            raise HTTPException(400, f'Missing {fc.factor} loading for one or more assets.')
        constraints.append({'type':'ineq','fun':lambda w,b=beta,lo=float(fc.lower): float(w@b-lo)})
        constraints.append({'type':'ineq','fun':lambda w,b=beta,hi=float(fc.upper): float(hi-w@b)})
    x0=np.ones(n)/n
    bounds=[(req.lower_bound,req.upper_bound) for _ in range(n)]
    res=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=constraints,options={'maxiter':2000,'ftol':1e-12})
    if not res.success:
        raise HTTPException(422, f'Risk parity optimization failed: {res.message}')
    w=np.maximum(res.x,0)
    w=w/w.sum()
    return {t:float(v) for t,v in zip(tickers,w)}

def _performance(weights: Dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float):
    w = np.array([weights[k] for k in mu.index], dtype=float)
    er = float(np.dot(w, mu.values))
    vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
    sharpe = float((er - rf) / vol) if vol > 0 else None
    return {'expected_return': er, 'volatility': vol, 'sharpe': sharpe}




def _exact_min_vol(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest):
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    _apply_factor_constraints(ef, req, list(mu.index))
    ef.min_volatility()
    weights = dict(ef.clean_weights())
    return {'weights': weights, 'performance': _performance(weights, mu, cov, rf)}


def _exact_max_sharpe(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest):
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    _apply_factor_constraints(ef, req, list(mu.index))
    if req.l2_gamma > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
    ef.max_sharpe(risk_free_rate=rf)
    weights = dict(ef.clean_weights())
    return {'weights': weights, 'performance': _performance(weights, mu, cov, rf)}


def _black_litterman_benchmark(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest):
    if not req.absolute_views:
        return None
    bad = [k for k in req.absolute_views if k not in mu.index]
    if bad:
        return None
    bl = BlackLittermanModel(cov, pi=mu, absolute_views=req.absolute_views)
    bl_mu = bl.bl_returns()
    bl_cov = bl.bl_cov()
    ef = EfficientFrontier(bl_mu, bl_cov, weight_bounds=bounds)
    _apply_factor_constraints(ef, req, list(bl_mu.index))
    if req.l2_gamma > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
    ef.max_sharpe(risk_free_rate=rf)
    weights = dict(ef.clean_weights())
    return {'weights': weights, 'performance': _performance(weights, bl_mu, bl_cov, rf)}


def _benchmarks(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest):
    out = {}
    try:
        out['min_volatility'] = _exact_min_vol(mu, cov, bounds, rf, req)
    except Exception as exc:
        out['min_volatility_error'] = str(exc)
    try:
        out['max_sharpe'] = _exact_max_sharpe(mu, cov, bounds, rf, req)
    except Exception as exc:
        out['max_sharpe_error'] = str(exc)
    try:
        rp = _risk_parity_weights(cov, req)
        out['risk_parity'] = {'weights': rp, 'performance': _performance(rp, mu, cov, rf)}
    except Exception as exc:
        out['risk_parity_error'] = str(exc)
    try:
        bl = _black_litterman_benchmark(mu, cov, bounds, rf, req)
        if bl is not None:
            out['black_litterman'] = bl
    except Exception as exc:
        out['black_litterman_error'] = str(exc)
    return out


def _constraint_diagnostics(weights: Dict[str, float], req: OptimizeRequest, tickers: List[str]):
    tol = 5e-4
    lower = float(req.lower_bound)
    upper = float(req.upper_bound)
    binding_lower = [t for t in tickers if abs(float(weights.get(t, 0.0)) - lower) <= tol]
    binding_upper = [t for t in tickers if abs(float(weights.get(t, 0.0)) - upper) <= tol]
    factor_rows = []
    for fc in req.factor_constraints:
        vals = np.array([float(fc.loadings[t]) for t in tickers], dtype=float)
        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
        exposure = float(w @ vals)
        binding = abs(exposure - float(fc.lower)) <= 1e-3 or abs(exposure - float(fc.upper)) <= 1e-3
        factor_rows.append({'factor': fc.factor, 'exposure': exposure, 'lower': float(fc.lower), 'upper': float(fc.upper), 'binding': bool(binding)})
    return {
        'solver_status': 'optimal',
        'lower_bound': lower,
        'upper_bound': upper,
        'binding_lower': binding_lower,
        'binding_upper': binding_upper,
        'factor_constraints': factor_rows,
    }

def _scipy_constraints(req: OptimizeRequest, tickers: List[str]):
    cons=[{'type':'eq','fun':lambda w: float(np.sum(w)-1.0)}]
    for fc in req.factor_constraints:
        beta=np.array([fc.loadings.get(t) for t in tickers],dtype=float)
        if np.isnan(beta).any():
            raise HTTPException(400, f'Missing {fc.factor} loading for one or more assets.')
        cons.append({'type':'ineq','fun':lambda w,b=beta,lo=float(fc.lower): float(w@b-lo)})
        cons.append({'type':'ineq','fun':lambda w,b=beta,hi=float(fc.upper): float(hi-w@b)})
    return cons


def _max_return_feasible(mu: pd.Series, bounds, req: OptimizeRequest) -> float:
    tickers=list(mu.index); n=len(tickers)
    x0=np.ones(n)/n
    bnds=[bounds for _ in range(n)]
    res=minimize(lambda w: -float(w@mu.values), x0, method='SLSQP', bounds=bnds,
                 constraints=_scipy_constraints(req,tickers),
                 options={'maxiter':2000,'ftol':1e-12})
    if not res.success:
        return float(mu.max())
    return float(res.x @ mu.values)


def _min_vol_point(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest):
    ef=EfficientFrontier(mu,cov,weight_bounds=bounds)
    _apply_factor_constraints(ef,req,list(mu.index))
    ef.min_volatility()
    ret,vol,sh=ef.portfolio_performance(risk_free_rate=rf)
    return {'return':float(ret),'volatility':float(vol),'sharpe':float(sh)}


def _frontier(mu: pd.Series, cov: pd.DataFrame, bounds, rf: float, req: OptimizeRequest, n_points: int = 45):
    """Return only the efficient (upper) branch using feasible target returns.

    The old implementation scanned from min(mu) to max(mu), which generated many
    infeasible/duplicated targets and could make the plotted curve look broken or
    compressed.  We now start at the constrained minimum-volatility portfolio's
    expected return and finish at the maximum feasible expected return.
    """
    try:
        minpt=_min_vol_point(mu,cov,bounds,rf,req)
        lo=float(minpt['return'])
        hi=_max_return_feasible(mu,bounds,req)
    except Exception:
        return []
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        return [minpt]
    pts=[]
    # Stay infinitesimally inside the upper feasibility boundary to avoid
    # numerical solver failures on the exact corner portfolio.
    targets=np.linspace(lo, lo + (hi-lo)*0.999, n_points)
    for target in targets:
        try:
            ef=EfficientFrontier(mu,cov,weight_bounds=bounds)
            _apply_factor_constraints(ef,req,list(mu.index))
            ef.efficient_return(float(target))
            ret,vol,sh=ef.portfolio_performance(risk_free_rate=rf)
            if np.isfinite(ret) and np.isfinite(vol):
                pts.append({'return':float(ret),'volatility':float(vol),'sharpe':float(sh)})
        except Exception:
            continue
    # Deduplicate near-identical solver points and enforce increasing volatility.
    out=[]
    for p in sorted(pts,key=lambda x:x['volatility']):
        if not out or abs(p['volatility']-out[-1]['volatility'])>1e-6 or abs(p['return']-out[-1]['return'])>1e-6:
            out.append(p)
    return out


@app.get('/health')
def health():
    return {'ok': True, 'engine': 'PyPortfolioOpt', 'version': '0.10.6', 'auth_required': bool(QUANT_SERVICE_SECRET)}


@app.post('/optimize')
def optimize(req: OptimizeRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    prices = _frame(req)
    returns = expected_returns.returns_from_prices(prices).dropna(how='any')
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    cov = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
    bounds = (req.lower_bound, req.upper_bound)
    weights: Dict[str, float]
    perf = None

    try:
        if req.method in {'max_sharpe','min_volatility','max_quadratic_utility','efficient_return','efficient_risk'}:
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
            _apply_factor_constraints(ef, req, list(mu.index))
            if req.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
            if req.method == 'max_sharpe': ef.max_sharpe(risk_free_rate=req.risk_free_rate)
            elif req.method == 'min_volatility': ef.min_volatility()
            elif req.method == 'max_quadratic_utility': ef.max_quadratic_utility(risk_aversion=req.risk_aversion)
            elif req.method == 'efficient_return':
                if req.target_return is None: raise HTTPException(400, 'target_return is required.')
                ef.efficient_return(req.target_return)
            elif req.method == 'efficient_risk':
                if req.target_volatility is None: raise HTTPException(400, 'target_volatility is required.')
                ef.efficient_risk(req.target_volatility)
            weights = dict(ef.clean_weights())
            perf = _performance(weights, mu, cov, req.risk_free_rate)

        elif req.method == 'black_litterman':
            bad_views = [k for k in req.absolute_views if k not in mu.index]
            if bad_views:
                raise HTTPException(400, f'Black-Litterman views contain unknown assets: {bad_views}')
            bl = BlackLittermanModel(cov, pi=mu, absolute_views=req.absolute_views or None)
            bl_mu = bl.bl_returns()
            bl_cov = bl.bl_cov()
            ef = EfficientFrontier(bl_mu, bl_cov, weight_bounds=bounds)
            _apply_factor_constraints(ef, req, list(bl_mu.index))
            if req.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=req.l2_gamma)
            ef.max_sharpe(risk_free_rate=req.risk_free_rate)
            weights = dict(ef.clean_weights())
            perf = _performance(weights, bl_mu, bl_cov, req.risk_free_rate)
            perf['prior'] = 'historical mean returns'
            perf['views'] = req.absolute_views
            perf['posterior_returns'] = {k: float(v) for k,v in bl_mu.items()}

        elif req.method == 'risk_parity':
            weights = _risk_parity_weights(cov, req)
            perf = _performance(weights, mu, cov, req.risk_free_rate)
            perf['risk_budget'] = req.risk_budgets or {t: 1.0/len(mu) for t in mu.index}

        elif req.method == 'hrp':
            if req.factor_constraints: raise HTTPException(400, 'Factor constraints are not supported for HRP.')
            opt = HRPOpt(returns)
            weights = dict(opt.optimize())
            perf = _performance(weights, mu, cov, req.risk_free_rate)

        elif req.method in {'cla_min_volatility','cla_max_sharpe'}:
            if req.factor_constraints: raise HTTPException(400, 'Factor constraints are not supported for CLA.')
            opt = CLA(mu, cov, weight_bounds=bounds)
            if req.method == 'cla_min_volatility': opt.min_volatility()
            else: opt.max_sharpe()
            weights = dict(opt.clean_weights())
            perf = _performance(weights, mu, cov, req.risk_free_rate)

        elif req.method == 'min_semivariance':
            opt = EfficientSemivariance(mu, returns, weight_bounds=bounds)
            _apply_factor_constraints(opt, req, list(mu.index))
            opt.min_semivariance()
            weights = dict(opt.clean_weights())
            perf = {'expected_return': float(np.dot(np.array([weights[k] for k in mu.index]), mu.values)), 'risk_measure': 'semivariance'}

        elif req.method == 'min_cvar':
            opt = EfficientCVaR(mu, returns, beta=req.cvar_beta, weight_bounds=bounds)
            _apply_factor_constraints(opt, req, list(mu.index))
            opt.min_cvar()
            weights = dict(opt.clean_weights())
            er, cvar = opt.portfolio_performance()
            perf = {'expected_return': float(er), 'cvar': float(cvar), 'beta': req.cvar_beta}

        elif req.method == 'min_cdar':
            opt = EfficientCDaR(mu, returns, beta=req.cdar_beta, weight_bounds=bounds)
            _apply_factor_constraints(opt, req, list(mu.index))
            opt.min_cdar()
            weights = dict(opt.clean_weights())
            er, cdar = opt.portfolio_performance()
            perf = {'expected_return': float(er), 'cdar': float(cdar), 'beta': req.cdar_beta}
        else:
            raise HTTPException(400, 'Unsupported optimization method.')

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'Optimization failed: {exc}')

    return {
        'engine': 'PyPortfolioOpt',
        'method': req.method,
        'observations': int(len(prices)),
        'data_start': str(prices.index.min().date()),
        'data_end': str(prices.index.max().date()),
        'weights': weights,
        'performance': perf,
        'expected_returns': {k: float(v) for k,v in mu.items()},
        'frontier': _frontier(mu, cov, bounds, req.risk_free_rate, req),
        'factor_exposure': _factor_exposure(weights, req.factor_constraints, list(mu.index)),
        'benchmarks': _benchmarks(mu, cov, bounds, req.risk_free_rate, req),
        'constraint_diagnostics': _constraint_diagnostics(weights, req, list(mu.index))
    }
