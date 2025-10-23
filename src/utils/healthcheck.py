from loguru import logger
import pathlib

def check_dirs():
    needed = ["data/raw", "data/clean", "data/features", "models", "reports"]
    for d in needed:
        p = pathlib.Path(d)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"OK: {p.resolve()}")

def main():
    logger.info("Healthcheck start")
    check_dirs()
    logger.info("Healthcheck done ✅")

if __name__ == "__main__":
    main()

