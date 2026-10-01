# PortfolioOPTIM v0.13.0 — Render deployment

Deploy the contents of `quant_service/` to the `FinancialStrategy/quant_service` repository.

Required Render environment variable:
- `QUANT_SERVICE_SECRET`

Start command is defined by the Dockerfile/render configuration. After deploy, verify:

`https://quant-service-ozki.onrender.com/health`

Expected identity:

```json
{"ok":true,"engine":"PortfolioOPTIM","version":"0.13.0","auth_required":true}
```

Netlify must use the same secret and:
- `PORTFOLIOOPTIM_API_URL=https://quant-service-ozki.onrender.com`
- `QUANT_SERVICE_SECRET=<same secret>`

The underlying Python library package names are implementation details and are not part of the public PortfolioOPTIM product identity.
