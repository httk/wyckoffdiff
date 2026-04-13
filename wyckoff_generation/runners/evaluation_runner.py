import hashlib
import json
import os

import pandas as pd
import wandb

from wyckoff_generation.common.registry import registry
from wyckoff_generation.common.utils import compare_hash
from wyckoff_generation.evaluation import compute_fwd
from wyckoff_generation.runners.base_runner import BaseRunner


@registry.register_runner("evaluate")
class EvaluationRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.get_eval_checkpoint()

    def init_model(self, config):
        pass

    def init_dataloaders(self, config):
        pass

    def init_optimizer(self, config):
        pass

    def load_checkpoint(self, config):
        pass

    def run(self):
        folder = self.compute_fwd(self.config)
        self.compute_prototype_stats(folder, self.config["num_samples_in_evaluation"])

    def get_eval_checkpoint(self):
        folder = "evaluation_checkpoint"
        checkpoint_file_path = os.path.join(folder, "checkpoint.pth")
        if not os.path.isfile(checkpoint_file_path):
            print("Wrenformer checkpoint not found, will download")
            os.makedirs(folder, exist_ok=True)
            run = wandb.Api().run("janosh/matbench-discovery/2kozbp4q")
            run.file("checkpoint.pth").download(root=folder)
            print("Downloaded")
        else:
            print("Wrenformer checkpoint already downloaded")
        assert compare_hash(
            checkpoint_file_path,
            "20cecd1560e5fc71851cf8675216ed7e30ccb53f18e86de6cfd8ffd090d35f36",
        ), f"Hash mismatch for Wrenformer checkpoint at {checkpoint_file_path}"

    def compute_fwd(self, args):
        folder, file = os.path.split(args["generated_datafile"])
        (
            results_dict,
            gen_data_subsampled_df,
            gen_data_novel_subsampled_df,
        ) = compute_fwd.main(args)
        with open(
            os.path.join(
                folder,
                f"statistics_num_samples={args['num_samples_in_evaluation']}.json",
            ),
            "w",
        ) as f:
            json.dump(results_dict, f, indent=4)
        columns_to_save = ["protostructures", "prototypes", "novel", "novel_prototype"]
        gen_data_subsampled_df.to_csv(
            os.path.join(
                folder,
                f"gen_data_subsampled_num_samples={args['num_samples_in_evaluation']}.csv",
            ),
            columns=columns_to_save,
            index=False,
        )
        gen_data_novel_subsampled_df.to_csv(
            os.path.join(
                folder,
                f"gen_data_novel_subsampled_num_samples={args['num_samples_in_evaluation']}.csv",
            ),
            columns=columns_to_save,
            index=False,
        )
        return folder

    def compute_prototype_stats(self, folder, num_samples):
        file = os.path.join(
            folder, f"gen_data_novel_subsampled_num_samples={num_samples}.csv"
        )
        df = pd.read_csv(file)
        assert len(df.index) == num_samples
        assert df["novel"].all()

        novel_prototype_df = df[df["novel_prototype"]]

        unique_prototypes = novel_prototype_df["prototypes"].unique().tolist()
        print("\n\nNumber of novel and unique prototypes: ", len(unique_prototypes))
