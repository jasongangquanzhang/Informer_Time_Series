# test_moirai_load.py

import os
import json
import torch
from huggingface_hub import hf_hub_download, snapshot_download

import timesfm


# Loading the timesfm-2.0 checkpoint:
def main():
    # For Torch
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=500,
            num_layers=50,
            use_positional_embedding=True,
            context_len=2048,
            point_forecast_mode='median'
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            # path="TimesFM/pretrained_models/torch_model.ckpt",
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch",
            local_dir="TimesFM/pretrained_models",
        ),
    )
    forecast, _ = tfm.forecast(
            inputs=[[1,2,3,4,5,6,7,8,9,10]],  # Example input
            freq=[2],                    # high-frequency
            window_size=None,            # no decomposition
            forecast_context_len=10,
            return_forecast_on_context=False,
            normalize=False,
        )
    print(f"Forecast: {forecast}")



# Loading the timesfm-1.0 checkpoint:


# # For Torch
# tfm = timesfm.TimesFm(
#       hparams=timesfm.TimesFmHparams(
#           backend="gpu",
#           per_core_batch_size=32,
#           horizon_len=128,
#       ),
#       checkpoint=timesfm.TimesFmCheckpoint(
#           huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
#   )
if __name__ == "__main__":
    try:
        main()
        print("✅ TimesFM model loaded successfully!")
    except Exception as e:
        print("❌ Failed to load TimesFM model:")
        print(e)
