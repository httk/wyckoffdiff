import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
from aviary.wren.utils import get_protostructure_label_from_spglib
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from wyckoff_generation.datasets.data_utils import build_protostructure

warnings.filterwarnings("ignore")


def cifstrings_to_protostructs(strings_list):
    aflow_labels = []
    for cif in strings_list:
        aflow_labels.append(
            get_protostructure_label_from_spglib(Structure.from_str(cif, "cif"))
        )
    df = pd.DataFrame(aflow_labels, columns=["wyckoff"])
    return df


def data_from_csv(file_path):
    df = pd.read_csv(file_path)
    if "wyckoff" in df.columns:
        pass
    elif "wyckoff_spglib" in df.columns:
        df["wyckoff"] = df["wyckoff_spglib"]
    elif "cif" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["wyckoff"] = cifstrings_to_protostructs(df["cif"])
    else:
        raise ValueError(
            "No column that is readable. Check input file, or implement function that creates protostructure"
        )

    return df[["wyckoff"]]


def _safe_build(d):
    try:
        return build_protostructure(d)
    except ValueError:
        print("Error with material, skipping")
        return None


def data_from_wyckoffdiff_list(data):
    st = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_safe_build, data))

    results = [r for r in results if r is not None]  # drop failed entries
    et = time.time()
    print(f"Total time for reading WyckoffDiff data: {et - st:.2f} s")
    return pd.DataFrame(results, columns=["wyckoff"])


def data_from_cdvae_dict(data):
    lengths = data["lengths"]
    if lengths.dim() == 3:  # CDVAE
        assert lengths.shape[0] == 1
        lengths = lengths.squeeze(0)
    angles = data["angles"].squeeze(0)
    num_atoms_per_material = data["num_atoms"].squeeze(0)
    frac_coords = data["frac_coords"].squeeze(0)
    frac_coords = torch.split(frac_coords, num_atoms_per_material.tolist())
    atom_types = data["atom_types"].squeeze(0)
    if atom_types.dim() == 2:  # SymmCD
        atom_types = torch.where(atom_types == 1)[1] + 1
    atom_types = torch.split(atom_types, num_atoms_per_material.tolist())
    aflow_labels = []
    for l, ang, frac, atms in zip(lengths, angles, frac_coords, atom_types):
        # Check for NaN, inf, or out-of-range values
        if (
            torch.isnan(l).any()
            or torch.isnan(ang).any()
            or torch.isnan(frac).any()
            or torch.isinf(l).any()
            or torch.isinf(ang).any()
            or torch.isinf(frac).any()
            or (l > 1e4).any()
        ):
            print("Skipping due to NaN or inf values")
            continue
        structure = Structure(
            Lattice.from_parameters(*l, *ang), atms, frac, coords_are_cartesian=False
        )
        protostructure = get_protostructure_label_from_spglib(structure)
        # Filter out NULL entries from spglib outputs
        if protostructure:
            aflow_labels.append(protostructure)
    df = pd.DataFrame(aflow_labels, columns=["wyckoff"])
    print(f"Effective data size: {df.shape[0]}")
    return df


def data_from_pt(file_path):
    data = torch.load(file_path, map_location="cpu")
    if isinstance(data, dict):
        return data_from_cdvae_dict(data)
    elif isinstance(data, list):
        return data_from_wyckoffdiff_list(data)
    else:
        ValueError(f"Unknown datatype of {file_path}: {type(data)}")


def get_dataset_df(file_path, num_samples=None):
    if file_path.endswith(".csv"):
        df = data_from_csv(file_path)
    elif file_path.endswith(".pt"):
        df = data_from_pt(file_path)
    else:
        raise ValueError("file extension not known")

    if num_samples is not None:
        if len(df.index) < num_samples:
            raise RuntimeError(
                f"The file {file_path} contains less than the requested {num_samples} materials"
            )
        if len(df.index) > num_samples:
            df = df.sample(
                n=num_samples, replace=False, ignore_index=True, random_state=42
            )
    return df
