# Desina

A Python prototype managed with [uv](https://docs.astral.sh/uv/).

## Getting started

Install uv, then run:

```sh
uv sync
uv run desina
```

The project uses Python 3.14. uv can download it automatically if needed.
You can also run the application with `uv run python -m desina`.

## Development

Application code lives in `src/desina/`.

```sh
uv add <package>          # Add a runtime dependency
uv add --dev <package>    # Add a development dependency
uv lock                  # Refresh the dependency lockfile
```

Commit `uv.lock` to keep environments reproducible.
