# Internal: Synthefy Package Deployment Guide

How to build and deploy the `synthefy` package on pypi.

## Prerequisites

1. **PyPI Account**: Ensure you have access to the Synthefy PyPI account
2. **API Token**: Generate a PyPI API token from your account settings
3. **UV Package Manager**: Ensure `uv` is installed in your environment

## Complete Working Workflow

Based on our tested environment, here's the complete sequence that works:

```bash
# 0. Make sure the __version__ in pyproject and __init__ is updated

# 1. Navigate to synthefy directory
cd synthefy-package/synthefy/

# 2. Set up virtual environment and install dependencies
uv sync
source .venv/bin/activate

# 3. Clean any previous builds
rm -rf dist/ build/ *.egg-info/

# 4. Set up environment variables (see Environment Variables section below)
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="<your-twine-api-key>"

# 5. Build the package
uv run -m build .

# 6. Upload to PyPI
uv run -m twine upload dist/*
```

## Troubleshooting

### Common Issues

1. **Version Already Exists**: PyPI doesn't allow duplicate versions. Update the version number.
2. **Build Errors**: Ensure all dependencies are properly specified in `pyproject.toml`.
3. **Upload Failures**: Check your PyPI credentials and ensure you have upload permissions.

## Package Structure

The package structure follows standard Python packaging conventions:
```
synthefy/
├── pyproject.toml          # Package configuration
├── src/
│   └── synthefy/          # Source code
│       ├── __init__.py     # Package initialization
│       ├── api_client.py   # API client implementation
│       └── data_models.py  # Pydantic models
├── tests/                  # Test suite
└── README.md              # Public documentation
```

## Testing (local vs dev vs prod)

We support running API-hitting tests against either:

- **prod (default)**: `https://forecast.synthefy.com`
- **dev**: `https://dev.forecast.synthefy.com`
- **local**: `http://localhost:{FORECASTING_API_PORT}`

This is controlled by a pytest option added in `tests/conftest.py`:

- `--synthefy-api-target=prod` (default)
- `--synthefy-api-target=dev`
- `--synthefy-api-target=local`

### Environment variables

- **Dev/Prod**
  - `SYNTHEFY_API_KEY`: required for tests that hit `synthefy.com`. If it is not set, those API-hitting tests will be **skipped**.
- **Local**
  - `FORECASTING_API_PORT`: optional (defaults to `8018`). Used to build the local base URL as `http://localhost:{FORECASTING_API_PORT}`.
  - No API key is required (and the client is configured with `api_key=None`).

### Run tests (prod)

```bash
cd synthefy-package/synthefy-forecasting/synthefy
export SYNTHEFY_API_KEY="..."
uv run pytest -q
```

### Run tests (dev)

```bash
cd synthefy-package/synthefy-forecasting/synthefy
export SYNTHEFY_API_KEY="..."
uv run pytest -q --synthefy-api-target=dev
```

### Run tests (local)

1) Bring up the local forecasting API (from the repo root):

```bash
docker compose -f synthefy-forecasting/docker-compose.yml up 
```

2) Run pytest targeting local:

```bash
cd synthefy-package/synthefy-forecasting/synthefy
export FORECASTING_API_PORT=8018
uv run pytest -q --synthefy-api-target=local
```

Notes:
- The tests set the SDK client `base_url` to `http://localhost:${FORECASTING_API_PORT}` when using `--synthefy-api-target=local`.
- If you prefer to run only the API integration tests, you can select them directly, e.g.:

```bash
uv run pytest -q tests/test_core_forecast_backtest_api.py --synthefy-api-target=local
```