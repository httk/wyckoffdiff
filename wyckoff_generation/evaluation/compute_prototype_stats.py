import os
import sys

import pandas as pd


def main(folder, num_samples):
    file = os.path.join(
        folder, f"gen_data_novel_subsampled_num_samples={num_samples}.csv"
    )
    df = pd.read_csv(file)
    assert len(df.index) == 10000
    assert df["novel"].all()

    novel_prototype_df = df[df["novel_prototype"]]
    # print(novel_prototype_df.head())

    unique_prototypes = novel_prototype_df["prototypes"].unique().tolist()
    print(len(unique_prototypes))
