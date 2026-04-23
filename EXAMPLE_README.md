# <Your Project Name>

## 1. Description

[**This section should provide a brief, high-level overview of the project.** Explain what it does, its main purpose, and the problem it solves. Keep it concise and easy for newcomers to understand.]

This repository serves as a template for [describe the kind of application, e.g., "a Python backend service"]. It includes a basic structure for configuration management, application logic, and deployment with Docker.

Brief note on structure and where code goes:
- Put your application code under `src/` (e.g., `src/app/main.py`).
- Use `src/config/` for configuration loading/merging logic.
- Keep shared helpers in `src/utils/` and tests in `tests/`.

## 2. Before Running

[**This section should describe what the user needs before running**.]
### 2.1. Prerequisites

You will need one of the following to run this project:
*   **UV:** A fast Python package installer and virtual environment manager.
*   **Docker:** A platform for developing, shipping, and running applications in containers.
    - **Recommended**: Docker Engine 20.10+ with Docker Compose plugin (v2.0+)
    - The newer `docker compose` command (with space) is preferred over the legacy `docker-compose`

### 2.2. Environment Variables

This project uses environment variables for configuration, especially for sensitive data like API keys. You should create a `.env` file in the root of the project for local development. This file should not be committed to version control.

To get started, copy the example file:
```bash
cp .env.example .env
```

Then, edit `.env` with your specific settings.

Here is the content for `.env.example`:
```
# Example environment variables.
# Copy this file to .env and fill in your actual values.

# API Keys
# SOME_API_KEY="your_api_key"

# Example of overriding config.default.yaml with environment variables
# APP__database__user="my_user"
# APP__database__password="my_password"
```

Where to place these: keep `.env` at repo root; see `SAFETY GUIDELINES` below for do-nots.

### 2.3. Configuration Files

The application uses a hierarchical configuration system that combines a default configuration file with local overrides and environment variables.

- `config.default.yaml`: This file contains default, non-sensitive configuration values. It should be committed to version control.
- `config.override.yaml`: This file can be used for local overrides. It is ignored by version control, so it's a safe place for machine-specific settings.
- **Environment Variables**: As mentioned above, environment variables can override settings from both YAML files. See `src/config/loader.py` for the exact loading mechanism.

[**This section should be updated to describe the specific configuration parameters your application uses.** Explain what each setting in `config.default.yaml` does.]

> IMPORTANT: Create the `config.override.yaml` file (it can be empty) before running.
>
> The loader expects this file to exist and merges it over `config.default.yaml`. If you don't have overrides yet, create an empty file:
> ```bash
> touch config.override.yaml
> ```

### 2.4. Safety Guidelines (Quick)
- No hardcoded secrets; use `.env` and environment variables.
- Commit only non-sensitive defaults (`config.default.yaml`); keep `config.override.yaml` private.


## 3. How to Run
[**In this section describe how to run your project**]
You can run this application using either `uv` or `Docker`.

### 3.1. Using UV

1.  Install dependencies:
    ```bash
    uv sync
    ```

2.  Run the application:
    ```bash
    uv run python -m <your_package_name>.main
    ```

### 3.2. Using Docker

#### Docker Compose
For a streamlined experience, you can use Docker Compose:
```bash
docker compose up
```

> IMPORTANT: Create a local override compose file for development.
>
> Copy the provided example to `docker-compose.override.yml` so that Docker Compose automatically picks it up:
> ```bash
> cp docker-compose.override.example.yml docker-compose.override.yml
> ```

To force rebuild (needed when dependencies change):
```bash
docker compose build --no-cache
docker compose up
```

> Note: Use `docker compose` (with a space) which is a docker plugin, is advisable instead of the standalone `docker-compose` (with a hyphen).

#### Classic Docker Build and Run
Alternatively, you can build and run the Docker container manually:

1.  Build the Docker image:
    ```bash
    docker build --no-cache -t <your_project_name> .
    ```
    
    > Note: Use `--no-cache` to ensure fresh builds when dependencies have changed.

2.  Run the Docker container:
    Make sure your `.env` file is present in the root directory.
    ```bash
    docker run --env-file .env <your_project_name>
    ```

## 4. Development Notes

When you modify dependencies in `pyproject.toml` (such as changing version constraints), you need to update the lock file:

```bash
# Update all dependencies to latest compatible versions
uv sync

# Update a specific package to its latest compatible version
uv sync --upgrade-package <package_name>

```

Important: Always update the lock file ( through sync command ) before building Docker images when dependencies have changed, otherwise Docker will use the cached versions from the old lock file.


