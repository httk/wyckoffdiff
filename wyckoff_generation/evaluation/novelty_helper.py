import numpy as np
import pandas as pd
from aviary.wren.utils import get_prototype_from_protostructure

from wyckoff_generation.datasets.data_utils import (
    compare_generated_with_training_dataset_fast_label_list,
)


def get_statistics_from_df(dataset_comparison):
    result_dict = {}

    # protostructures
    protostructures = list(dataset_comparison["protostructures"])
    unique_protostructures = set(protostructures)
    result_dict["uniqueness"] = len(unique_protostructures) / len(protostructures)
    novel_array = np.array(dataset_comparison["novel"])
    result_dict["novelty"] = novel_array.mean()
    novel_protostructures = list(dataset_comparison["protostructures"][novel_array])
    novel_unique_protostructures = list(set(novel_protostructures))
    result_dict["novel_uniqueness"] = len(novel_unique_protostructures) / len(
        novel_protostructures
    )
    result_dict["novel_and_unique_protostructures"] = novel_unique_protostructures

    # prototypes
    prototypes = list(dataset_comparison["prototypes"])
    unique_prototypes = set(prototypes)
    result_dict["uniqueness_prototypes"] = len(unique_prototypes) / len(prototypes)
    novel_prototype_array = np.array(dataset_comparison["novel_prototype"])
    result_dict["novel_prototypes"] = novel_prototype_array.mean()
    novel_prototypes = list(dataset_comparison["prototypes"][novel_prototype_array])
    novel_unique_prototypes = set(novel_prototypes)
    result_dict["novel_prototypes_uniqueness"] = len(novel_unique_prototypes) / len(
        novel_prototypes
    )
    result_dict["novel_and_unique_prototypes"] = novel_unique_protostructures
    return result_dict


def get_enriched_df(gen_data_df, train_df):
    gen_data_list = [
        [i, row["wyckoff"], get_prototype_from_protostructure(row["wyckoff"])]
        for i, row in gen_data_df.iterrows()
    ]

    return compare_generated_with_training_dataset_fast_label_list(
        gen_data_list, list(train_df["wyckoff"]), "training"
    )
