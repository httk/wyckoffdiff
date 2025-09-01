import argparse
import json
import os
import sys
from datetime import datetime

import torch

default_args_dict = {
    # General
    "seed": 42,
    "dataset": "wbm",
    # Training
    "optimizer": "AdamW",
    "lr": 2e-4,
    "batch_size": 256,
    "l2_reg": 0.0,
    "epochs": 1000,
    "val_interval": 1,
    "num_workers": 0,
    # D3PM
    "t_max": 1000,
    "num_elements": 100,
    "max_num_atoms": 54,
    "d3pm_transition": "uniform",
    "loss_fn": "ce",
    "gnn": "base",
    # GNN
    "num_gnn_layers": 3,
    "hidden_dim": 256,
    "dof_pos_sg_emb_size": 16,
    "gnn_activation": "SiLU",
    # MLP
    "mlp_hidden_layers": 2,
    "mlp_activation": "SiLU",
    # Generation
    "num_samples": 10000,
}


def get_parser():
    """
    Function to obtain the config-dictionary with all parameters to be used in training/testing
    :return: config - dictionary with all parameters to be used
    """
    parser = argparse.ArgumentParser(
        description="Generation of Wyckoff positions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # If config file should be used
    parser.add_argument(
        "--config", type=str, help="Config file to read run config from"
    )

    # General
    parser.add_argument(
        "--seed",
        type=int,
        default=default_args_dict["seed"],
        help="Seed for random number generator",
    )
    parser.add_argument("--mode", type=str.lower, help="What mode to run in")
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Whether to disable cuda, even if available",
    )
    parser.add_argument("--logger", type=str.lower, help="Logger type (none or wandb)")

    parser.add_argument("--load", type=str, help="Path to model parameters to load")
    parser.add_argument(
        "--dataset",
        type=str.lower,
        default=default_args_dict["dataset"],
        help="Dataset name",
    )

    # Tranining
    parser.add_argument(
        "--optimizer",
        type=str,
        default=default_args_dict["optimizer"],
        help="What type of optimizer (case-sensitive, should correspond to how it is called in Pytorch",
    )
    parser.add_argument(
        "--lr", type=float, default=default_args_dict["lr"], help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args_dict["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=default_args_dict["l2_reg"],
        help="L2 regularization factor",
    )

    # TODO: Move these from here and make specific objects where they will be used return them?
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_args_dict["epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=default_args_dict["val_interval"],
        help="Evaluation interval, number of gradient steps",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_args_dict["num_workers"],
        help="num_workers for dataloader",
    )

    parser.add_argument(
        "--pin_memory", type=bool, default=True, help="pin_memory passed to dataloader"
    )

    parser.add_argument(
        "--persistent_workers",
        type=bool,
        default=True,
        help="persistent_workers passed to dataloader in case num_workers>0",
    )

    parser.add_argument(
        "--backbone", type=str.lower, help="What type of backbone model"
    )
    parser.add_argument(
        "--model", type=str.lower, help="What type of sampling model (e.g., our D3PM)"
    )

    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    # D3PM
    parser.add_argument(
        "--t_max",
        type=int,
        default=default_args_dict["t_max"],
        help="Largest timestep, T, when sampling",
    )
    parser.add_argument(
        "--num_elements",
        type=int,
        default=default_args_dict["num_elements"],
        help="Number of elements",
    )
    parser.add_argument(
        "--max_num_atoms",
        type=int,
        default=default_args_dict["max_num_atoms"],
        help="Maximum number of atoms of a specific element at a specific wyckoff position",
    )
    parser.add_argument(
        "--d3pm_transition",
        type=str.lower,
        default=default_args_dict["d3pm_transition"],
        help="What type of D3PM transition",
    )

    parser.add_argument(
        "--loss_fn",
        type=str.lower,
        default=default_args_dict["loss_fn"],
        help="What type of loss in D3PM",
    )

    parser.add_argument(
        "--gnn",
        type=str.lower,
        default=default_args_dict["gnn"],
        help="What type of GNN in D3PM",
    )

    parser.add_argument(
        "--hybrid_lambda",
        type=float,
        help="Lambda for upweighting CE loss when computing hybrid D3PM loss (i.e., loss = KL + lambda*CE)",
    )

    parser.add_argument(
        "--verbose_losses",
        action="store_true",
        help="Verbose training losses if desired during training.",
    )

    # GNN
    parser.add_argument(
        "--num_gnn_layers",
        type=int,
        default=default_args_dict["num_gnn_layers"],
        help="Number of GNN layers",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=default_args_dict["hidden_dim"],
        help="Hidden dimension of node vectors",
    )

    parser.add_argument(
        "--dof_pos_sg_emb_size",
        type=int,
        default=default_args_dict["dof_pos_sg_emb_size"],
        help="Dimension of dof- position- and spacegroup embedding",
    )

    parser.add_argument(
        "--gnn_activation",
        type=str,
        default=default_args_dict["gnn_activation"],
        help="GNN activation function",
    )

    parser.add_argument(
        "--no_multiplicity_encoding",
        action="store_true",
        help="Do not encode multiplicity (for backward compability)",
    )

    parser.add_argument(
        "--binary_dof_encoding",
        action="store_true",
        help="Use binary encoding of DoF (0 or not 0 DoF, for backward compability)",
    )

    parser.add_argument(
        "--no_softmax",
        action="store_true",
        help="Do not use softmax in GNN layer (for backward compability)",
    )

    # MLPs
    parser.add_argument(
        "--mlp_hidden_layers",
        type=int,
        default=default_args_dict["mlp_hidden_layers"],
        help="Number of hidden MLP layers",
    )

    parser.add_argument(
        "--mlp_activation",
        type=str,
        default=default_args_dict["mlp_activation"],
        help="MLP activation function",
    )

    # Generation
    parser.add_argument(
        "--num_samples",
        type=int,
        default=default_args_dict["num_samples"],
        help="Number of samples to generate",
    )

    # Post processing of generated samples
    parser.add_argument(
        "--post_process_all",
        action="store_true",
        help="Post process all unprocessed data checkpoints. Delete previous post_processed directory to re-run. For specific additions run with --load <path_to_data_dir> instead, can also be to already created post processing file.",
    )
    parser.add_argument(
        "--enrich_data",
        action="store_true",
        help="Enrich generated data set with functions provided in data_utils.py.",
    )
    parser.add_argument(
        "--use_processed_data",
        action="store_true",
        help="Identifies novelty of the generated data. Requires protostructure to be defined.",
    )
    parser.add_argument(
        "--novelty_reference_set",
        type=str,
        help="Can be 'train', 'val', or 'test' for WBM.",
    )
    parser.add_argument(
        "--save_protostructures",
        action="store_true",
        help="Use to save protostructures and prototypes in addition to updating the dataset. Will be saved in the output directory.",
    )

    return parser


def get_included_config(file_path):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)
        if "include" in config:
            included_config = get_included_config(config["include"])
            assert "include" not in included_config
            included_config.update(config)
            del included_config["include"]
            config = included_config.copy()
    return config


def get_config():
    # First parse, and include default values
    default_parser = get_parser()
    default_args = default_parser.parse_args()
    config = vars(default_args)

    # Now update with arguments from config file
    if default_args.config:
        assert os.path.exists(default_args.config), f"No config file: {default_args}"
        with open(default_args.config) as json_file:
            config_from_file = json.load(json_file)
        if (
            "include" in config_from_file
        ):  # ability to include another json to avoid having to specify multiple things
            included_config = get_included_config(config_from_file["include"])
            included_config.update(config_from_file)
            config_from_file = included_config.copy()
            del config_from_file["include"]
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(
            ["Unknown option in config file: {}".format(opt) for opt in unknown_options]
        )
        assert not unknown_options, unknown_error
        config.update(config_from_file)

    # Now parse again, but without any default values
    command_line_parser = argparse.ArgumentParser(
        parents=[default_parser], add_help=False
    )
    command_line_parser.set_defaults(**{key: None for key in config.keys()})
    command_line_args = command_line_parser.parse_args()

    # now overwrite values in config with those from command line
    flags_from_command_line = []
    config_from_command_line = vars(command_line_args)
    for key, value in config_from_command_line.items():
        if value is not None:
            if key == "config":
                value = str(value)
                value = os.path.splitext(value)[0]
                value = "-".join(os.path.normpath(value).split(os.path.sep))
            if key != "logger":
                if key == "load":
                    # Reduce loadstring to date time only for reference.
                    split_value = value.split("/")
                    assert (
                        "checkpoints" in split_value
                    ), "Ensure that load parameters is in checkpoints directory."
                    for idx, element in enumerate(split_value):
                        if element == "checkpoints":
                            #
                            prev_run_name = split_value[idx + 1]
                            string_value = prev_run_name.split("_")[0]
                else:
                    string_value = value
                flags_from_command_line.append(
                    f"{key}={string_value}" if key != "config" else str(string_value)
                )
            config[key] = value

    config["cuda"] = not config["disable_cuda"] and torch.cuda.is_available()
    if "SLURM_JOB_ID" in os.environ:
        config["job_id"] = os.environ.get("SLURM_JOB_ID")
    else:
        config["job_id"] = None
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    config["run_name"] = "_".join([dt_string] + flags_from_command_line)
    if not config["mode"].startswith("train") or config["debug"]:
        config["logger"] = "none"
        print(
            "Not running any training/in debug mode, therefore using no-op logger",
            file=sys.stdout,
        )
    return config


if __name__ == "__main__":
    print(get_config(), file=sys.stdout)
