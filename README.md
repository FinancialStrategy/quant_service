# MK PortfolioOPTIM Quant Service — v0.13.0

Secure Python analytical backend for MK Institutional Investment Intelligence. Netlify serves the public application and proxies requests to this service through `PORTFOLIOOPTIM_API_URL`; browser clients do not require the backend URL or secret.

## Core optimization
Maximum Sharpe / Tangency, Minimum Volatility, Efficient Return/Risk, Quadratic Utility, Black–Litterman, Risk Parity, HRP, CLA, Minimum Semivariance, Minimum CVaR and Minimum CDaR.

## v0.13 institutional extensions
- Expected-return model selector: historical mean, EMA, CAPM proxy.
- Covariance selector: Ledoit–Wolf, sample, EWMA.
- Factor/group/region/asset-class constraints, minimum Effective N, turnover, tracking error and liquidity capacity.
- Common-model risk decomposition and constraint diagnostics.
- Portfolio-level walk-forward/OOS endpoint.
- Robustness/model-grid/bootstrap endpoint with resampled exact-frontier envelope.
- Black–Litterman absolute/relative views, confidence and tau.

## Endpoints
- `GET /health`
- `POST /optimize`
- `POST /walkforward`
- `POST /robustness`

Run locally:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
