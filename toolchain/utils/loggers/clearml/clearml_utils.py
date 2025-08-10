"""Main Logger class for ClearML experiment tracking."""

import glob
import re
from pathlib import Path

import yaml

try:
    import clearml
    from clearml import Dataset, Task

    assert hasattr(clearml, "__version__")
except (ImportError, AssertionError):
    clearml = None

def construct_dataset(clearml_info_string):
    """Load in a clearml dataset and fill the internal data_dict with its contents."""
    dataset_id = clearml_info_string.replace("clearml://", "")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_root_path = Path(dataset.get_local_copy())

    yaml_filenames = list(
        glob.glob(str(dataset_root_path / "*.yaml")) + glob.glob(str(dataset_root_path / "*.yml"))
    )
    if len(yaml_filenames) > 1:
        raise ValueError(
            "More than one yaml file was found in the dataset root, cannot determine which one contains "
            "the dataset definition this way."
        )
    elif len(yaml_filenames) == 0:
        raise ValueError(
            "No yaml definition found in dataset root path, check that there is a correct yaml file "
            "inside the dataset root path."
        )
    with open(yaml_filenames[0]) as f:
        dataset_definition = yaml.safe_load(f)

    assert set(dataset_definition.keys()).issuperset({"train", "test", "val", "nc", "names"}), (
        "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"
    )

    data_dict = dict()
    data_dict["train"] = (
        str((dataset_root_path /
             dataset_definition["train"]).resolve()) if dataset_definition["train"] else None
    )
    data_dict["test"] = (
        str((dataset_root_path /
             dataset_definition["test"]).resolve()) if dataset_definition["test"] else None
    )
    data_dict["val"] = (
        str((dataset_root_path /
             dataset_definition["val"]).resolve()) if dataset_definition["val"] else None
    )
    data_dict["nc"] = dataset_definition["nc"]
    data_dict["names"] = dataset_definition["names"]

    return data_dict

class ClearmlLogger:
    """Log training runs, datasets, models, and predictions to ClearML.

    This logger sends information to ClearML at app.clear.ml or to your own hosted server. By default,
    this information includes hyperparameters, system configuration and metrics, model metrics, code information and
    basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    """
    def __init__(self, opt, hyp):
        """
        - Initialize ClearML Task, this object will capture the experiment
        - Upload dataset version to ClearML Data if opt.upload_dataset is True

        arguments:
        opt (namespace) -- Commandline arguments for this run
        hyp (dict) -- Hyperparameters for this run

        """
        self.current_epoch = 0

        self.current_epoch_logged_images = set()

        self.bbox_interval = opt.bbox_interval
        self.clearml = clearml
        self.task = None
        self.data_dict = None
        if self.clearml:
            self.task = Task.init(
                project_name=opt.project if opt.project != "runs/train" else "YOLOv5",
                task_name=opt.name if opt.name != "exp" else "Training",
                tags=["YOLOv5"],
                output_uri=True,
                auto_connect_frameworks={"pytorch": False},
            )

            self.task.connect(hyp, name="Hyperparameters")

            if opt.data.startswith("clearml://"):
                self.data_dict = construct_dataset(opt.data)

                opt.data = self.data_dict

    def log_debug_samples(self, files, title="Debug Samples"):
        """
        Log files (images) as debug samples in the ClearML task.

        arguments:
        files (List(PosixPath)) a list of file paths in PosixPath format
        title (str) A title that groups together images with the same values
        """
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                self.task.get_logger().report_image(
                    title=title,
                    series=f.name.replace(it.group(), ""),
                    local_path=str(f),
                    iteration=iteration,
                )
