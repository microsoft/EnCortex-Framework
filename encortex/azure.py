import json
import logging
import os

from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

import typer

logger = logging.getLogger(__name__)


def get_config():
    encortex_dir_path = os.path.join(os.path.expanduser("~"), ".encortex")
    encortex_config_path = os.path.join(encortex_dir_path, "config.encortex")
    if not os.path.exists(encortex_config_path):
        raise FileNotFoundError(
            "EnCortex config not configured. Please run encortex_setup to configure the Azure environment"
        )

    with open(encortex_config_path, "r") as f:
        config = json.load(f)
        return config


def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
            return False
    except:
        return False
    else:  # pragma: no cover
        return True


def run_script_on_aml(
    compute_cluster_name: str,
    environment_name: str,
    script_dir_path: str,
    script_name: str,
    experiment_name: str,
) -> None:
    encortex_config = get_config()

    assert (
        "aml_config" in encortex_config.keys()
    ), "No AML config.json path found. Please run encortex_setup again"
    if encortex_config["aml_config"] == "":
        logging.error("No AML config.json found, exiting.")
        exit(0)
    else:
        assert (
            "docker_image" in encortex_config.keys()
        ), "No Docker Image registry link found"

    # Setup the workspace
    ws = Workspace.from_config(path=encortex_config["aml_config"])

    # Initialize the compute and experiment objects
    cluster = ComputeTarget(workspace=ws, name=compute_cluster_name)
    env = ws.environments[environment_name]

    src = ScriptRunConfig(
        source_directory=script_dir_path,
        script=script_name,
        environment=env,
        compute_target=cluster,
    )

    run = Experiment(ws, experiment_name).submit(src)

    if is_notebook():
        try:
            from azureml.widgets import RunDetails

            RunDetails(run).show()
        except:
            logger.error(
                "Couldn't find azureml.widgets module. Please install it and try again to visualize the run on your notebook"
            )


def main():
    typer.run(run_script_on_aml)


if __name__ == "__main__":
    typer.run(main)
