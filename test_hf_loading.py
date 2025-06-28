# test_moirai_load.py

import os
import json
import torch
from huggingface_hub import hf_hub_download, snapshot_download

# Optional: clear HF mirror config
os.environ.pop("HF_ENDPOINT", None)

# Load config
config_path = hf_hub_download(
    repo_id="Salesforce/moirai-1.1-R-base",
    filename="config.json",
    local_files_only=False
)
with open(config_path) as f:
    config = json.load(f)

# Map the distribution output name to an actual object
# You'll need to replace this with the real class
# For example, from gluonts.torch.distributions import StudentTOutput

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule, MoiraiFinetune

# Instantiate model
print("🔧 Instantiating MoiraiModule...")
module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-base")