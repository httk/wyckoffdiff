import bz2
import hashlib
import os
import os.path as osp
import sys

import aviary.wren.data as aviary_wren_data
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader

from wyckoff_generation.common.registry import registry
from wyckoff_generation.datasets.lookup_tables import (
    element_number,
    spg_wyckoff,
    spg_wyckoff_degrees_of_freedom,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)


def extract_wyckoff_data_and_properties(data_frame_row: pd.DataFrame):
    # Use aviary protostrucutre parser
    (
        spg,
        _,  # Add multiplicities if we want them on the aviary form
        elements,
        wyckoff_set,
    ) = aviary_wren_data.parse_protostructure_label(data_frame_row["wyckoff_spglib"])

    # Extract desired properties
    if "e_form_per_atom_wbm" in data_frame_row:
        e_form_per_atom = data_frame_row["e_form_per_atom_wbm"]
    elif "formation_energy_per_atom" in data_frame_row:
        e_form_per_atom = data_frame_row["formation_energy_per_atom"]
    elif "energy_per_atom" in data_frame_row:
        e_form_per_atom = data_frame_row["energy_per_atom"]
    else:
        raise ValueError(
            "Seems like dataset does not have energy information, or the name of the column is not defined here"
        )

    # Collect into new dataframe
    data_dict = {
        "aflow_label": data_frame_row["wyckoff_spglib"],
        "space_group": spg,
        # "multiplicities": multiplicities, # Uncomment if we want to add multiplicities on aviary form.
        "elements": elements,
        "wyckoff_set": wyckoff_set,
        "e_form_per_atom": e_form_per_atom,
    }
    df = pd.Series(data_dict)

    return df


def map_element_to_index(element: str):
    # Each index corresponds to the atom number in the periodic table. 0 Means no atom.
    return element_number.get(element)


def map_wyckoff_label_to_index(wyckoff_label: str):
    # Wyckoff letter to number.
    # Change to zero-based indexing
    return wyckoff_label_to_index.get(wyckoff_label) - 1


def index_to_wyckoff_label(idx: int):
    # Get Wyckoff label from index
    wyckoff_label_list = wyckoff_label_to_index.keys()
    return wyckoff_label_list[idx]


def matrix_list(rows: int, cols: int, value: int):
    # Initialize a matrix shaped list with initial elements set to value.
    return [[value for _ in range(cols)] for _ in range(rows)]


def map_nested_list(func, lst: list):
    # Map function for nested lists
    if all(isinstance(l, list) for l in lst) or all(isinstance(l, tuple) for l in lst):
        return [map_nested_list(func, l) for l in lst]
    else:
        return list(map(func, lst))


def format_wyckoff_element_matrix(extracted_wyckoff_row: pd.DataFrame):

    # ``elements`` contains chemical formula of each atom, size n.
    # ``wyckoff_sets`` contains tuples of length n, where each tuple holds the wyckoff site/position for the corresponding element on the same index in ``elements``.
    elements = extracted_wyckoff_row["elements"]
    wyckoff_sets = extracted_wyckoff_row["wyckoff_set"]
    space_group = extracted_wyckoff_row["space_group"]

    # Empty list to store wem matrices
    wem_list = []

    # Map labels and elements to indices
    # print(f"Elements: {elements}.")
    # print(f"wyckoff_sets: {wyckoff_sets}.")
    element_indices = map_nested_list(map_element_to_index, elements)
    wyckoff_set_indices = map_nested_list(map_wyckoff_label_to_index, wyckoff_sets)

    # print(f"ElementIndices: {element_indices}.")
    # print(f"WyckoffIndices: {wyckoff_set_indices}.\n")

    # Fetch degrees of freedom dict
    wyckoff_dof = spg_wyckoff_degrees_of_freedom.get(space_group)

    # Set matrix rows with Wyckoff positions without degrees of freedom to 0
    # for wyckoff_site, degrees_of_freedom in wyckoff_dof.items():
    #     if degrees_of_freedom == 0:
    #         wyckoff_index = map_wyckoff_label_to_index(wyckoff_site)
    #         wem[wyckoff_index] = 0

    # Add 1 to each matrix element on the indices of corresponding element/atom type and wyckoff site.
    for wyckoff_indices, wyckoff_set in zip(wyckoff_set_indices, wyckoff_sets):
        # Initialize empty matrix to store matrix encoding.
        wyckoff_position_dimension = len(spg_wyckoff.get(str(space_group)))
        element_dimension = len(element_number) + 1
        wem = matrix_list(wyckoff_position_dimension, element_dimension, 0)
        # wem = np.zeros((wyckoff_position_dimension, element_dimension))
        if len(wyckoff_indices) == len(element_indices):
            for wyckoff_site_index, element_index, wyckoff_pos in zip(
                wyckoff_indices, element_indices, wyckoff_set
            ):
                # TODO: Could iterate here and skip mapping above when we need this many, in that case can fetch atom number directly.
                dof = wyckoff_dof.get(wyckoff_pos)

                if dof == 0:
                    # For wyckoff positions without degrees of freedom only atom number for the corresponding atom is on the entire row.
                    # +1 to get the corresponding atom number.
                    wem[wyckoff_site_index][0] = element_index
                else:
                    # If degrees of freedom is not zero
                    wem[wyckoff_site_index][element_index] = (
                        wem[wyckoff_site_index][element_index] + 1
                    )

            # Add the equivalent wem to list
            wem_list.append(wem)
        else:
            raise ValueError(
                f"Number of wyckoff sites '{len(wyckoff_indices)}' is not equeal to the corresponding number of elements (atom types) '{len(element_indices)}'."
            )

    return wem_list


def fetch_wyckoff_degrees_of_freedom(space_group: str) -> list:
    """
    Returns a list of wyckoff sites for the corresponding 'space_group'. The list is ordered with wyckoff site 'a' as the first element.
    """

    # Reverse list to get wyckoff site 'a' first.
    dof = list(spg_wyckoff_degrees_of_freedom.get(space_group).values())
    dof.reverse()

    return dof


def fetch_wyckoff_multiplicities(space_group: str) -> list:
    """
    Returns a list of multiplicities for each wyckoff site for the corresponding 'space_group'. The list is ordered with multiplicity for wyckoff site 'a' as the first element, indexed by the corresponding wyckoff site.
    """

    # Reverse list to get multiplicity for wyckoff site 'a' first.
    multiplicities = list(spg_wyckoff_multiplicities.get(space_group).values())
    multiplicities.reverse()

    return multiplicities


def preprocess(raw_file_path) -> pd.DataFrame:
    if raw_file_path.endswith(".csv"):
        wd_df = pd.read_csv(raw_file_path)
    else:
        raise ValueError("Only supporting '.csv' raw files.")

    print(wd_df["wyckoff_spglib"], file=sys.stdout)

    parsed_aflow_labels = wd_df.apply(
        lambda row: extract_wyckoff_data_and_properties(row), axis=1
    )
    parsed_aflow_labels["wyckoff_element_matrix"] = parsed_aflow_labels.apply(
        lambda row: format_wyckoff_element_matrix(row), axis=1
    )
    parsed_aflow_labels["degrees_of_freedom"] = parsed_aflow_labels.apply(
        lambda row: fetch_wyckoff_degrees_of_freedom(row["space_group"]), axis=1
    )
    parsed_aflow_labels["multiplicities"] = parsed_aflow_labels.apply(
        lambda row: fetch_wyckoff_multiplicities(row["space_group"]), axis=1
    )

    # print(parsed_aflow_labels.head)
    # print(parsed_aflow_labels.columns)

    return parsed_aflow_labels


def compare_hash(data_path, correct_hash):
    sha256_hash = hashlib.sha256()
    with open(data_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == correct_hash


def decompress_bz2_file(
    compressed_file_path, remove_original=False, decompressed_file_path=None
):
    # Get the file name without the .bz2 extension
    if decompressed_file_path is None:
        if compressed_file_path.endswith(".bz2"):
            decompressed_file_path = compressed_file_path[:-4]
        else:
            decompressed_file_path = compressed_file_path + ".decompressed"

    # Open the compressed file and read its contents
    with bz2.open(compressed_file_path, "rb") as source:
        # Open a new file to write the decompressed data
        with open(decompressed_file_path, "wb") as dest:
            dest.write(source.read())

    # Remove the original compressed file if requested
    if remove_original:
        os.remove(compressed_file_path)
        print(
            f"Decompressed {compressed_file_path} to {decompressed_file_path} and removed the original file"
        )
    else:
        print(f"Decompressed {compressed_file_path} to {decompressed_file_path}")

    return decompressed_file_path


class WyckoffDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split,
        num_elements=118,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        debug=False,
    ):
        self.base_url = f"https://public.openmaterialsdb.se/wyckoffdiff/data/{self.suffix}/raw/"  # splits for WyckoffDiff paper. Overwrite if using own hosting
        self.debug = debug
        self.split = split
        self.num_elements = num_elements
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.debug:
            return [f"wyckoff_data_{self.suffix}_debug.pt"]
        return [f"wyckoff_data_{self.suffix}_{self.split}.pt"]

    @property
    def download_name(self):
        return self.raw_file_names[0] + ".bz2"

    def download(self):
        zipped_file = self.download_name
        url = self.base_url + zipped_file
        download_url(url, self.raw_dir)

        zipped_file = osp.join(self.raw_dir, zipped_file)

        if not compare_hash(zipped_file, self.hash_dict[self.split]):
            raise ValueError(
                f"Invalid hash for downloaded file: {self.suffix} {self.split} set. To redownload, remove the file: {zipped_file}"
            )

        decompress_bz2_file(
            zipped_file,
            remove_original=True,
            decompressed_file_path=osp.join(self.raw_dir, self.raw_file_names[0]),
        )

    def wyckoff_data_to_graph(self, data_row: pd.Series) -> Data:

        # print('wyckoff_element_matrix', data_row['wyckoff_element_matrix'])
        # print("space_group", int(data_row["space_group"]))
        # print("e_form_per_atom_wbm", data_row["e_form_per_atom_wbm"])
        # print("aflow_label", data_row["aflow_label"])
        # print("elements", data_row["elements"])
        # print("wyckoff_set", data_row["wyckoff_set"])
        # print("multiplicities", data_row["multiplicities"])
        stacked_wem_tensors = torch.tensor(data_row["wyckoff_element_matrix"])
        stacked_wem_tensors = stacked_wem_tensors.flatten(0, 1)

        num_pos = len(data_row["degrees_of_freedom"])
        e_i = torch.arange(num_pos).repeat_interleave(num_pos)
        e_j = torch.arange(num_pos).repeat(num_pos)
        edge_index = torch.stack([e_i, e_j])

        wyckoff_pos_idx = torch.arange(num_pos)
        degrees_of_freedom = torch.tensor(data_row["degrees_of_freedom"])
        zero_dof = degrees_of_freedom == 0
        num_0_dof = torch.sum(zero_dof)
        num_inf_dof = torch.sum(~zero_dof)

        data = Data(
            # atom_list in this case is either atomic number or list of atom number encoded indices in a list.
            x=stacked_wem_tensors,
            edge_index=edge_index,
            space_group=torch.tensor(int(data_row["space_group"])),
            # one type of formation energy. A property we may want later.
            e_form_per_atom=torch.tensor(data_row["e_form_per_atom"]),
            # These are kept from raw input
            aflow_label=data_row["aflow_label"],
            elements=data_row["elements"],
            wyckoff_set=data_row["wyckoff_set"],
            multiplicities=torch.tensor(data_row["multiplicities"]),
            degrees_of_freedom=degrees_of_freedom,
            wyckoff_pos_idx=wyckoff_pos_idx,
            num_pos=torch.tensor([num_pos]),
            zero_dof=zero_dof,
            num_0_dof=num_0_dof,
            num_inf_dof=num_inf_dof,
            # elements=torch.tensor(data_row["elements"]),
            # wyckoff_set=torch.tensor(data_row["wyckoff_set"]),
            # multiplicities=torch.tensor(data_row["multiplicities"])
        )

        return data

    def process(self):
        print("Extracting and converting aflow labels...", file=sys.stdout)
        wyckoff_data_df = preprocess(osp.join(self.raw_dir, self.raw_file_names[0]))
        # Read data into huge `Data` list.
        print("Converting wyckoff data to graph...", file=sys.stdout)
        data_list = [
            self.wyckoff_data_to_graph(row) for _, row in wyckoff_data_df.iterrows()
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"Saving to '{self.processed_paths[0]}'...", file=sys.stdout)
        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

    def get(self, idx: int):
        data = super().get(idx)
        # In chase of multiple equivalent Wyckoff sets (i.e., same material), we randomly return one of them
        # This is ok for training, but might cause problems if dataset is used for other purpose
        data.x = torch.chunk(data.x, len(data.wyckoff_set), 0)[
            torch.randint(len(data.wyckoff_set), (1,))
        ]

        return Data(
            x_0_dof=data.x[data.zero_dof, 0],
            x_inf_dof=data.x[~data.zero_dof, 1 : (self.num_elements + 1)],
            edge_index=data.edge_index,
            space_group=data.space_group,
            multiplicities=data.multiplicities,
            num_pos=data.num_pos,
            num_nodes=data.num_pos,
            zero_dof=data.zero_dof,
            num_0_dof=data.num_0_dof,
            num_inf_dof=data.num_inf_dof,
            wyckoff_pos_idx=data.wyckoff_pos_idx,
            degrees_of_freedom=data.degrees_of_freedom,
        )

    @classmethod
    def get_dataloaders(cls, config):
        loaders = []
        for split in ["train", "val", "test"]:
            dataset = cls(
                split=split, num_elements=config["num_elements"], debug=config["debug"]
            )
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=config["batch_size"],
                    shuffle=split == "train",
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                    persistent_workers=(
                        config["persistent_workers"] and config["num_workers"] > 0
                    ),
                )
            )
        return loaders


@registry.register_dataset("wbm")
class WBMDataset(WyckoffDataset):
    suffix = "wbm"
    hash_dict = {
        "train": "a0898053c5c124484d97030738e25e0567d33896c3be0a65b1bc22e701807ca6",
        "val": "61afbb7ea3031b269a3ac5bcd5499078e1d83480b6a214d3185c4e4797382ec3",
        "test": "f6aea31b07148f8d9e6f91e84d0a474aa830b4fa70d09f053e476f1822b2bae9",
    }

    def __init__(self, split, num_elements, debug=False):
        super().__init__(osp.join("data", "wbm"), split, num_elements, debug=debug)

    @property
    def raw_file_names(self):
        if self.debug:
            return ["2024-01-24-wbm-summary_debug.csv"]
        return [f"wbm_{self.split}.csv"]


@registry.register_dataset("mp20")
class MP20Dataset(WyckoffDataset):
    suffix = "mp20"
    hash_dict = {
        "train": "aa9ccf4cf83a58e2aba3492e12f0eaf75bdf2946a7c90fa5bbeea94d731b721a",
        "val": "6b73a81761a74ccf19eb066d4e2cc39a3aae9a0d149d61bd0f966fbf6584a230",
        "test": "716a2dfef434e03e43ac94d195729d4fd8ee9677f3fd11918bb2895e6c8be13a",
    }

    def __init__(self, split, num_elements, debug=False):
        super().__init__(osp.join("data", "mp20"), split, num_elements, debug=debug)

    @property
    def raw_file_names(self):
        if self.debug:
            raise ValueError("MP20 does not support debug mode")
        return [f"mp20_{self.split}.csv"]


@registry.register_dataset("carbon24")
class Carbon24Dataset(WyckoffDataset):
    suffix = "carbon24"
    hash_dict = {
        "train": "fcee223e4e99939a2600a710faae4e26c7bfcf087192735129ba53f85851e693",
        "val": "8ce62b2977d9f172e26376255a0496ac281e6d113a572f6cd8d150ac22730442",
        "test": "df5995fa0dce33b65ec1fe983e5b68e1fb2bacdcee882285052ac4ef08f38096",
    }

    def __init__(self, split, num_elements, debug=False):
        super().__init__(osp.join("data", "carbon24"), split, num_elements, debug=debug)

    @property
    def raw_file_names(self):
        if self.debug:
            raise ValueError("Carbon24 does not support debug mode")
        return [f"carbon24_{self.split}.csv"]
