import logging
import mlflow
from utils.parameters import get_parameters
from utils.io import load_pickle, save_pickle
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parameters = get_parameters("build_final_model")

    with mlflow.start_run(experiment_id=parameters["publish_experiment_id"]):
        logger.info("Loading sources")
        pipelines_paths = parameters["pipelines"]
        model_path = parameters["model"]
        output_path = parameters["output_path"]

        logger.info("Building final pipeline")
        pipelines = [(path, load_pickle(path)) for path in pipelines_paths]
        model = [(model_path, load_pickle(model_path))]
        steps = pipelines + model
        final_pipeline = Pipeline(steps=steps)

        logger.info(f"Saving final pipeline to {output_path}")
        save_pickle(final_pipeline, output_path)

        logger.info("Publishing artifact")
        mlflow.log_artifact(output_path)


if __name__ == "__main__":
    main()
