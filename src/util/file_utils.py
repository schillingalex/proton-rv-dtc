import os
import pathlib
import shutil

from util.config import RunConfig


def prepare_run_directory(run_config: RunConfig):
    """
    Preparation step of the working directory for an evaluation run.

    If the workdir does not exist, it is created.
    If the workdir exists and the run config indicates to purge, the directory is deleted and re-created.

    Afterward, the workdir is checked for an existing configuration file, which can exist there, if workdir exists
    and it was not purged. If the existing config does not match the new config, a ValueError is raised.

    Lastly, the configuration is dumped to a JSON file.

    :param run_config: The configuration used for the run, containing the workdir to use.
    """
    workdir_path = pathlib.Path(run_config.workdir)
    if run_config.purge_workdir:
        shutil.rmtree(run_config.workdir, ignore_errors=True)

    workdir_path.mkdir(parents=True, exist_ok=True)

    config_path = os.path.join(run_config.workdir, "config.json")
    if os.path.exists(config_path):
        old_config = RunConfig.from_file(config_path)
        if run_config != old_config:
            raise ValueError("Trying to use an existing workdir with a different configuration.")

    run_config.to_file(config_path)
