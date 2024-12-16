import os
import sys
import warnings

import matplotlib
import pyrootutils

matplotlib.use("Agg")  # https://github.com/microsoft/vscode-python-debugger/issues/205
pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=True)

import torch  # noqa: E402 need to be imported after pyrootutils.setup_root because of #thread env vars

from src.utils import CustomLightningCLI  # noqa: E402

warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    if os.environ.get("DEBUG", False):
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    cli = CustomLightningCLI(save_config_kwargs=dict(overwrite=True))
