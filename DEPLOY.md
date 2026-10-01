# MK Quant Optimizer deployment

The frontend remains on Netlify. The Python service can run on Railway, Render, or any Docker-compatible host.

## Required service environment variable
- `QUANT_SERVICE_SECRET`: create a long random secret. The `/optimize` endpoint requires it when configured.

## Netlify environment variables
- `PORTFOLIOOPTIM_API_URL`: public base URL of this service, without `/optimize`.
- `QUANT_SERVICE_SECRET`: same secret as the Python service.

The browser calls only `/api/optimizer`. Netlify adds the bearer secret server-side.

## Render free test route
Create a Web Service from `quant_service/`, choose Docker and Free, set `QUANT_SERVICE_SECRET`, deploy, then copy the service URL into Netlify `PORTFOLIOOPTIM_API_URL`.

## Railway route
Deploy `quant_service/` using its Dockerfile, set `QUANT_SERVICE_SECRET`, then copy its public service URL into Netlify `PORTFOLIOOPTIM_API_URL`.
