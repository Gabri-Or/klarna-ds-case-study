import logging

from klarna_ds_case_study.training.pipeline import run_pipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()

    run_pipeline()

    logging.getLogger(__name__).info("Pipeline complete.")


if __name__ == "__main__":
    main()
