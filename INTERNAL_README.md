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