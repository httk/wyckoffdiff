import sys
from collections import defaultdict

import aviary.wren.data as aviary_wren_data
import aviary.wren.utils as aviary_wren_utils
import pandas as pd
import torch
from aviary.wren.utils import (
    canonicalize_element_wyckoffs,
    cry_sys_dict,
    get_prototype_formula_from_composition,
)
from pymatgen.core import Composition
from pyxtal import Group
from torch_geometric.data import Data

from wyckoff_generation.datasets.lookup_tables import (
    element_number,
    spg_wyckoff_multiplicities,
    wyckoff_label_to_index,
)


def load_dataset(path_to_data_file: str) -> list[Data]:

    """
    PyTorch data set loader, wrapper to map to the correct device.

    Parameters
    ----------
        path_to_data_file : str
            String path to the PyTorch data set to load.

    Returns
    -------
        PyTorch dataset loaded.
    """

    if torch.cuda.is_available():
        return torch.load(path_to_data_file)
    else:
        return torch.load(path_to_data_file, map_location=torch.device("cpu"))


def get_wyckoff_labels_and_elements(
    wyckoff_element_matrix: torch.tensor,
) -> tuple[list, list, dict]:

    """
    Parse a matrix representation of wyckoff positions and elements used in WyckoffDiff.

    Parameters
    ----------
        wyckoff_element_matrix : torch.tensor
            Matrix representation of wyckoff-position and elements produced by WyckoffDiff

    Returns
    -------
        elements : list
            List of elements according to aviary format.
        wyckoff_labels : list
            List of tuple of wyckoff set.
        element_wyckoff_dict : list
            Dictionary representation of a structure, containing elements as keys and values are a list of tuples containing the corresponding wyckoff sites the element occupies. The length of the wyckoff sites tuple is the number of atoms located at the site.
    """

    # Reverse order in look-up tables
    wyckoff_index_to_label = {
        idx: label for label, idx in wyckoff_label_to_index.items()
    }
    number_to_element = {idx: element for element, idx in element_number.items()}

    wyckoff_indices, element_indices = torch.nonzero(
        wyckoff_element_matrix, as_tuple=True
    )

    if len(element_indices) == 0:
        raise ValueError(f"No elements found in element matrix. Skipping...")

    wyckoff_labels = tuple()
    elements = []
    element_wyckoff_dict = {}
    for wyckoff_index, element_index in zip(
        wyckoff_indices.tolist(), element_indices.tolist()
    ):
        wyckoff_label = wyckoff_index_to_label[
            wyckoff_index + 1
        ]  # +1 to convert to 1-based indexing

        if element_index == 0:
            # Zero degrees of freedom so just fetch the atom number
            element_num = wyckoff_element_matrix[wyckoff_index][element_index]
            element = number_to_element[int(element_num)]
            element_list = [element]
            # If no degrees of freedom we always have one atom
            number_of_atoms = 1
        else:
            # Inf degrees of freedom.
            # Get the element and multiply with the amount of atoms at the corresponding site.
            number_of_atoms = int(wyckoff_element_matrix[wyckoff_index][element_index])
            element = number_to_element[element_index]
            element_list = [element] * number_of_atoms

        # Update wyckoff label tuple
        wyckoff_label_multiplied = tuple(wyckoff_label) * number_of_atoms

        # Add new wyckoff labels to the resulting tuple, one label for each element
        wyckoff_labels += wyckoff_label_multiplied

        # Add new elements list
        elements += element_list

        # Update the elements_wyckoff dictionary
        prev_wyckoffs = element_wyckoff_dict.get(element)
        if prev_wyckoffs is None:
            element_wyckoff_dict[element] = [wyckoff_label_multiplied]
        else:
            # Add the new tuple to the list of previous tuples
            element_wyckoff_dict[element] = prev_wyckoffs + [wyckoff_label_multiplied]

    return elements, list(wyckoff_labels), element_wyckoff_dict


def assemble_protostructure(
    element_wyckoff_dict: dict, space_group: int, form: str = "aflow-label"
):

    """
    Assemble a protostructure from a ``element_wyckoff_dict`` parsed by ``get_wyckoff_labels_and_elements``.

    Parameters
    ----------
        element_wyckoff_dict : dict
            Dictionary representation of a structure, containing elements as keys and values are a list of tuples containing the corresponding wyckoff sites the element occupies. The length of the wyckoff sites tuple is the number of atoms located at the site.
        space_group : int
            Space group for the structure.
        form : str, default 'aflow-label'
            Format of the output protostructure, default is AFLOW format.
    Returns
    -------
        res_string : str
            A protostructure formatted according to the ``form``.

    See Also
    --------
    get_wyckoff_labels_and_elements : Parsing a matrix representation of wyckoff-position and elements
    """

    element_wyckoff_dict = dict(
        sorted(element_wyckoff_dict.items(), key=lambda x: x[0])
    )

    # Format the protostructure to the desired format
    if form == "aflow-label":

        all_elements = element_wyckoff_dict.keys()
        concat_elements = "-".join(all_elements)

        all_wyckoffs = element_wyckoff_dict.values()

        all_wyckoffs_multiplied = []
        # all_wyckoff is a list of tuples, where each tuple contains wyckoff labels of length: amount of atoms at the site, for the corresponding element.
        for wls_tuple in all_wyckoffs:
            wls_res = ""
            for wls in wls_tuple:
                amount = len(wls)
                wls_res += f"{str(amount)}{wls[0]}"
            all_wyckoffs_multiplied.append(wls_res)

        concat_wyckoff_sites = "_".join(all_wyckoffs_multiplied)
        # The canonicalize_element_wyckoffs funciton fetches the canonicalized representation of the wyckoff labels, thus it can change the provided letters to the one with the lowest score.
        # This allows us to directly compare canonicalized protostructures.
        concat_wyckoff_sites = canonicalize_element_wyckoffs(
            concat_wyckoff_sites, space_group
        )

        # Partial aflow_label
        partial_aflow_label = f"<chemform>_<pearson>_{space_group}_{concat_wyckoff_sites}:{concat_elements}"
        composition_dict = get_composition_dictionary_from_partial_aflow(
            partial_aflow_label
        )
        comp = Composition(composition_dict)

        pearson_symbol = get_pearson(space_group, comp)
        reduced_stochiometry = get_prototype_formula_from_composition(comp)

        res_string = f"{reduced_stochiometry}_{pearson_symbol}_{space_group}_{concat_wyckoff_sites}:{concat_elements}"

    return res_string


def get_pearson(spg, composition):
    gr = Group(int(spg))
    cry_sys_name = cry_sys_dict[gr.lattice_type]
    spg_sym = gr.symbol
    centering = "C" if spg_sym[0] in ("A", "B", "C", "S") else spg_sym[0]
    num_sites_conventional = composition.num_atoms
    pearson_symbol = f"{cry_sys_name}{centering}{num_sites_conventional}"

    return pearson_symbol


def get_composition_dictionary_from_partial_aflow(partial_aflow: str):
    _, _, elements, wyckoff_set = aviary_wren_data.parse_protostructure_label(
        partial_aflow
    )
    wyckoff_set = wyckoff_set[0]

    spg = partial_aflow.split("_")[2]
    group = spg_wyckoff_multiplicities[spg]

    # Composition dict holds elements and the number of atoms for each element including the multiplicity.
    composition_dict = defaultdict(int)
    for element, letter in zip(elements, wyckoff_set):
        composition_dict[element] += group[letter]

    return dict(composition_dict)


def build_protostructure(data_point: Data, form: str = "aflow-label"):

    """
    Parse and assemble a matrix representation of wyckoff positions and elements used in WyckoffDiff into a protostructure representation.

    Parameters
    ----------
        data_point : torch_geometric.data.Data
            One data point in a generated data set from WyckoffDiff.
        form : str
            Format of the output protostructure, default is AFLOW format.
    Returns
    -------
        res_string : str
            A protostructure formatted according to the ``form``.
    See Also
    --------
    get_wyckoff_labels_and_elements : Parsing a matrix representation of wyckoff-position and elements.
    assemble_protostructure : Assemble a parsed matrix representation of wyckoff-position and elements into a protostructure.
    """

    # Extract wyckoff labels and elements from the data matrix
    wyckoff_element_matrix = data_point.x

    _, _, element_wyckoff_dict = get_wyckoff_labels_and_elements(wyckoff_element_matrix)

    # Spacegroup
    spg = int(data_point.space_group)

    # Protostructure string
    res_string = assemble_protostructure(element_wyckoff_dict, spg, form=form)

    return res_string


def enrich_dataset(
    dataset: list[Data], return_protostructure_list: bool = False
) -> list[Data]:

    """
    Enrich a WyckoffDiff generated dataset with ``aflow_label`` a protostructure string, list of ``elements``, a list of all wyckoff labels and equivalent sets ``wyckoff_set``, and a prototype string ``canonical_prototype``.

    Parameters
    ----------
        dataset : list(torch_geometric.data.Data)
            A data set generated by WyckoffDiff.

    Returns
    -------
        dataset : list(torch_geometric.data.Data)
            Updated dataset, now containing ``aflow_label`` a protostructure string, list of ``elements``, a list of all wyckoff labels and equivalent sets ``wyckoff_set``, and a prototype string ``canonical_prototype``.
    """

    print(f"Enriching dataset...", file=sys.stdout)

    print(
        f"Building protostructures, saving to attribute 'aflow_label'...",
        file=sys.stdout,
    )

    all_aflow_labels = []
    all_prototype_labels = []

    for data_point in dataset:
        try:
            aflow_label = build_protostructure(data_point)
            data_point.aflow_label = aflow_label
            _, _, elements, wyckoff_set = aviary_wren_data.parse_protostructure_label(
                aflow_label
            )
            all_aflow_labels.append(aflow_label)
            data_point.elements = elements
            data_point.wyckoff_set = wyckoff_set
            canonical_prototype = aviary_wren_utils.get_prototype_from_protostructure(
                aflow_label
            )
            data_point.canonical_prototype = canonical_prototype
            all_prototype_labels.append(canonical_prototype)
        except ValueError as e:
            print(
                f"Could not create aflow label when enriching data. Error: {e}. \nSkip setting attribute in: {data_point}"
            )
            continue

    if return_protostructure_list:
        return dataset, all_aflow_labels, all_prototype_labels

    return dataset


def compare_generated_with_training_dataset_fast_label_list(
    generated_dataset: list[list[int, str, str]],
    training_dataset: list[str],
    set_name: str,
    return_duplicate_names: bool = False,
) -> pd.DataFrame:

    """

    generated_dataset: list of lists, where second level contains [original_index, protostructure_label, prototype_label]
    training_dataset: list of protostructure labels from reference set (e.g., train/val/test)
    set_name: name of dataset, include split for explicitness
    return_duplicate_names: return the duplicate names if needed for other function

    returns: pd.DataFrame containing protostructures, prototypes, novel, novel_prototype, duplicates_{set_name}_set", duplicates_{set_name}_set_prototype
    """

    print(
        f"Identifying novelty of generated data compared with {set_name} dataset, saving to attribute 'novel' and 'duplicates_{set_name}_set'...",
        file=sys.stdout,
    )

    # Create a dataframe from the aflow label lists and canonical prototypes in the generated dataset
    gen_df = pd.DataFrame(
        generated_dataset, columns=["original_index", "protostructures", "prototypes"]
    )

    # --- Protostructures and prototypes matching

    train_df = pd.DataFrame(training_dataset, columns=["protostructures"])
    train_df["prototypes"] = train_df.protostructures.apply(
        aviary_wren_utils.get_prototype_from_protostructure
    )

    # Check if novel (note that novel is opposite to is-in)
    gen_df["novel"] = ~gen_df.protostructures.isin(train_df.protostructures)
    gen_df["novel_prototype"] = ~gen_df.prototypes.isin(train_df.prototypes)

    # Collect duplicates
    # Protostructures
    duplicate_attribute_name = f"duplicates_{set_name}_set"
    train_grouped_protostructures = (
        train_df.groupby("protostructures")["protostructures"]
        .apply(list)
        .reset_index(name=duplicate_attribute_name)
    )
    gen_df = gen_df.merge(
        train_grouped_protostructures,
        how="left",
        left_on="protostructures",
        right_on="protostructures",
    )
    gen_df[duplicate_attribute_name] = gen_df[duplicate_attribute_name].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Prototypes
    duplicate_prototype_attribute_name = f"duplicates_{set_name}_set_prototype"
    train_grouped_prototypes = (
        train_df.groupby("prototypes")["prototypes"]
        .apply(list)
        .reset_index(name=duplicate_prototype_attribute_name)
    )
    gen_df = gen_df.merge(
        train_grouped_prototypes,
        how="left",
        left_on="prototypes",
        right_on="prototypes",
    )
    gen_df[duplicate_prototype_attribute_name] = gen_df[
        duplicate_prototype_attribute_name
    ].apply(lambda x: x if isinstance(x, list) else [])

    if return_duplicate_names:
        return gen_df, duplicate_attribute_name, duplicate_prototype_attribute_name

    return gen_df
