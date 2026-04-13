import torch
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer
from tqdm import tqdm

from wyckoff_generation.evaluation import read_file_utils
from wyckoff_generation.evaluation.frechet_distance import (
    frechet_distance_from_embeddings,
)
from wyckoff_generation.evaluation.novelty_helper import (
    get_enriched_df,
    get_statistics_from_df,
)


def get_embeddings(model, dataset):
    store = []

    def hook_fn(module, input, output):
        store.append(output)
        return output

    target_layer = list(model.children())[-2]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # ids_list = []
    with torch.no_grad():
        for d in dataset:
            (padded_features, mask, equivalence_counts), targets, ids = d
            # ids_list.extend(ids.tolist())
            output = model(padded_features, mask, equivalence_counts)
    hook_handle.remove()

    return torch.cat(store, dim=0)


def main(args):
    train_data_df_full = read_file_utils.get_dataset_df(
        args["train_datafile"],
    )
    print("Parsing generated materials")
    gen_data_df_full = read_file_utils.get_dataset_df(args["generated_datafile"])

    assert len(train_data_df_full.index) >= args["num_samples_in_evaluation"], len(
        train_data_df_full.index
    )
    assert len(gen_data_df_full.index) >= args["num_samples_in_evaluation"], len(
        gen_data_df_full.index
    )

    gen_data_df_enriched = get_enriched_df(gen_data_df_full, train_data_df_full)
    gen_data_df_subsampled = gen_data_df_enriched.sample(
        n=args["num_samples_in_evaluation"],
        replace=False,
        ignore_index=True,
        random_state=42,
    )
    train_data_df_subsampled = train_data_df_full.sample(
        n=args["num_samples_in_evaluation"],
        replace=False,
        ignore_index=True,
        random_state=42,
    )

    gen_data_fwd = df_to_in_mem_dataloader(
        gen_data_df_subsampled,
        input_col="protostructures",
        batch_size=args["batch_size"],
        shuffle=False,
        device=args["device"],
    )
    train_data_fwd = df_to_in_mem_dataloader(
        train_data_df_subsampled,
        input_col="wyckoff",
        batch_size=args["batch_size"],
        shuffle=False,
        device=args["device"],
    )

    state_dict = torch.load("evaluation_checkpoint/checkpoint.pth", map_location="cpu")
    model = Wrenformer(**state_dict["model_params"]).to(args["device"])
    model.load_state_dict(state_dict["model_state"])
    model.train(False)
    assert not model.training
    print("Computing training embeddings")

    # to improve stability, use double precision
    print("Computing Wrenformer embeddings of generated and training materials")
    train_embeddings = get_embeddings(model, train_data_fwd).double()
    gen_embeddings = get_embeddings(model, gen_data_fwd).double()
    fwd = float(frechet_distance_from_embeddings(train_embeddings, gen_embeddings))

    stats_subsampled = get_statistics_from_df(gen_data_df_subsampled)
    stats_subsampled["fwd"] = fwd
    print("\n\n----Stats for generated materials----")
    for key, value in stats_subsampled.items():
        if isinstance(value, float):
            print(f"{key}: {value}")

    gen_data_novel_only = gen_data_df_enriched.loc[gen_data_df_enriched["novel"]]
    assert len(gen_data_novel_only.index) >= args["num_samples_in_evaluation"]
    gen_data_novel_subsampled = gen_data_novel_only.sample(
        n=args["num_samples_in_evaluation"],
        replace=False,
        ignore_index=True,
        random_state=42,
    )
    gen_data_novel_fwd = df_to_in_mem_dataloader(
        gen_data_novel_subsampled,
        input_col="protostructures",
        batch_size=args["batch_size"],
        shuffle=False,
        device=args["device"],
    )

    gen_novel_embeddings = get_embeddings(model, gen_data_novel_fwd).double()
    fwd_novel = float(
        frechet_distance_from_embeddings(train_embeddings, gen_novel_embeddings)
    )
    stats_novel = get_statistics_from_df(gen_data_novel_subsampled)
    stats_novel["fwd"] = fwd_novel
    print("\n\n----Stats for generated novel materials----")
    for key, value in stats_novel.items():
        if isinstance(value, float):
            print(f"{key}: {value}")

    result_string = " & ".join(
        [
            f"{stats_subsampled['fwd']:.2f}",
            f"{stats_subsampled['novelty']*100:.1f}",
            f"{stats_subsampled['uniqueness']*100:.1f}",
            f"{stats_novel['fwd']:.2f}",
            f"{stats_novel['uniqueness']*100:.1f}",
        ]
    )
    print("\n\n----Results string for LaTeX table----\n", result_string)

    full_results_dict = {
        "stats_subsampled": {
            key: value
            for key, value in stats_subsampled.items()
            if isinstance(value, float)
        },
        "stats_novel": {
            key: value for key, value in stats_novel.items() if isinstance(value, float)
        },
        "result_string": result_string,
    }
    return full_results_dict, gen_data_df_subsampled, gen_data_novel_subsampled
