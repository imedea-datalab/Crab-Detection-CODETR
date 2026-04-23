from src.config.loader import load_config


def run() -> None:
    config = load_config()
    print(f"App started. DB host: {config['database']['host']}")


if __name__ == "__main__":
    run()


