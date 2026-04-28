## Guide for How to Use This Repository.

This project follows a secure-by-default template to prevent leakage of secrets, etc. Read and follow these rules before committing code.

### Good practices
- **Never hardcode secrets** (API keys, passwords, tokens) in code or config.
- Check `.gitignore` covers `.env`, `config.local.*`, secrets and other things which you don't want to commit.
- README is there and explains local setup and overrides.
- **Use environment variables** at runtime. Local variables go in `.env` (ignored by Git). See `.env.example`.
- **Base + Override pattern**:
  - `config.default.yaml` is safe and committed.
  - `config.override.yaml` is developer-specific and ignored by Git. You can also override from .env based on your preferences.
  - Loader merges default → override → `APP__...` environment variables.


### Configuration Files

There are 3 possible ways you can put your confugiration information for your project. The application uses a hierarchical configuration system ( Meaning that same config information provided in differnt places can override others ) that combines a default configuration file with local overrides or overrrides from environment variables - 
- `config.default.yaml`: This file contains default, non-sensitive configuration values. It should be committed to version control. This is the lowest priority.
- `config.override.yaml`: This file can be used for local overrides. It is ignored by version control, so it's a safe place for local-specific settings.
> IMPORTANT: Create the `config.override.yaml` file (it can be empty) before running. Because the loader expects this file to exist and merges it over `config.default.yaml`. If you don't have overrides yet, create an empty file:
> ```bash
> touch config.override.yaml
> ```
- **Environment Variables**: As mentioned above, environment variables can override settings from both YAML files( default and overrides ). here you just add `APP__...` prefix to the variables specified in your YAML. See `.env.example` on how to do it and See `src/config/loader.py` for the exact loading mechanism.
  - Use environment variables for configuration, especially for sensitive data like API keys. Create a `.env` ( looking at .env.example ) file in the root of the project for local development. This file should not be committed to version control. For production the env vars should be stored remotely (like in GitHub Actions, AWS, GCP, or a bare-metal server) and not in .env.

  To get started, copy the example file:
  ```bash
  cp .env.example .env
  ```
  Then, edit `.env` with your specific settings. Delete the `.env.example`. 
  - Env vars can override config via prefix `APP__` with `__` as nesting.
    Example: `APP__database__host=127.0.0.1` sets `database.host` in the YAML.
- If you are thinking that both .env and .override file can override, then which one to use to ovverride the default config. The correct way to do it is throug .env files. And .ovverrides should be avoided, but they also work. The difference in them is that in .env you have your overrides and sensitive info at the same place and in config.ovverride file they are at differnt places.

### Docker

- **Docker Compose**: `docker-compose.yml` references a env var file; a `docker-compose.override.example.yml` shows local dev options. Do not commit the docker overrides.
  - The values from the .env files will be fed to the right side (e.g., `${DB_HOST}`) of the env vars defined in the `docker-compose.yml` file. This is what Docker Compose looks for outside the container (on your host machine or in a .env file) to find the actual secret or value. The left side names of the env vars like `APP__database__host` are availaible inside the docker container as env vars and `loader.py` is designed to use them and make them availaible to your code.
  - In production the the secrets are injected into the host system's environment where the docker compose command is executed, not theough `.env` file. 
  - If a variable isn't found in either the .env file or the host environment, Docker Compose will leave it blank, which can crash your app. You can provide "fallback" or "default" values right inside the `docker-compose.yml` file using this syntax: `${VAR_NAME:-default_value}`.


### UV
`uv` is a new Python package installer and resolver, designed to be a faster drop-in replacement for `pip` and `pip-tools`. It is recommended to use `uv` for all package management operations.
- **Order of execution**: 
1. First enter the repo and do `uv init`. 
  - `uv init` creates a `.venv` folder and a `pyproject.toml` file.
2. Then after that for each addition of the dependency, do `uv pip install <dependency>`. For example, `uv pip install fastapi`. 
  - `uv pip install <dependency>` installs the dependency into the `.venv` and adds it to `pyproject.toml`.
  - You can also change the dependencies in the `pyprojects.toml` and then use `uv sync` to update the packages.
  - `uv` keeps your packages and whole venv contained in your project repo itself.