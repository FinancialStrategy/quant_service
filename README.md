# MK PyPortfolioOpt Quant Service

This service is intentionally separated from the Netlify front-end/runtime. Netlify serves the public application and proxies optimization requests to a Python service configured with `PYPORTFOLIOOPT_API_URL`.

Supported methods: max Sharpe, minimum volatility, quadratic utility, efficient return/risk, HRP, CLA, minimum semivariance, minimum CVaR and minimum CDaR.

Run locally:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
