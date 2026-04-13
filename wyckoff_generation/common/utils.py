"""
Parts of this file, the functions ``setup_imports``, ``_get_project_root``, and ``_import_local_file`` , are based on code made available under the MIT license by FAIR Chemistry available from their repository as https://github.com/facebookresearch/fairchem/blob/dbaefaed40eee2844d033c78ccd2fe68976ebcb6/ocpmodels/common/utils.py
Modifications Copyright (c) 2025 The High-Throughput Toolkit

The original copyright header from the fairchem file follows below:
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import hashlib
import importlib
import os
import re
import sys
from pathlib import Path

import torch


def get_pretrained_checkpoint(load_path, best=True):
    print("loading, ", load_path, file=sys.stdout)
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    return checkpoint


def compare_hash(data_path, correct_hash):
    sha256_hash = hashlib.sha256()
    with open(data_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == correct_hash


def increment_filename(file_path):
    # Split the file path into directory, filename, and extension
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    # Regex pattern to match a filename that ends with '(number)'
    pattern = re.compile(r"(.*)\((\d+)\)$")

    match = pattern.match(name)

    if match:
        base_name = match.group(
            1
        ).strip()  # Get the base part of the filename without the (number)
        number = int(match.group(2))  # Get the current number
    else:
        base_name = name
        number = 0  # Start with 0, then we'll add 1 later

    new_file_path = file_path  # Initial new file path as the original

    while os.path.exists(new_file_path):
        number += 1
        new_name = f"{base_name}({number}){ext}"  # Generate new filename with incremented number
        new_file_path = os.path.join(directory, new_name)  # Full path with new name

    return new_file_path


def save_generated_samples(samples_batch, dir, filename=None):
    # save a batch of samples into
    if filename is None:
        filename = "generated_samples.pt"
    full_path = os.path.join(dir, filename)
    full_path = increment_filename(full_path)
    torch.save(samples_batch.to_data_list(), full_path)
    return


def setup_imports(config: dict | None = None) -> None:
    from wyckoff_generation.common.registry import registry

    # skip_experimental_imports = (config or {}).get("skip_experimental_imports", False)
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        project_root = _get_project_root()
        # logging.info(f"Project root: {project_root}")
        importlib.import_module("wyckoff_generation.common.logging")

        import_keys = ["runners", "datasets", "models", "loggers"]
        for key in import_keys:
            for f in (project_root / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)

        # if not skip_experimental_imports:
        #    setup_experimental_imports(project_root)
    finally:
        registry.register("imports_setup", True)


def _get_project_root() -> Path:
    """
    Gets the root folder of the project
    :return: The absolute path to the project root.
    """
    from wyckoff_generation.common.registry import registry

    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("wyckoff_generation_core_root", no_warning=True)

    if root_folder is not None:
        assert isinstance(
            root_folder, str
        ), "wyckoff_generation_core_root must be a string"
        root_folder = Path(root_folder).resolve().absolute()
        assert root_folder.exists(), f"{root_folder} does not exist"
        assert root_folder.is_dir(), f"{root_folder} is not a directory"
    else:
        root_folder = Path(__file__).resolve().absolute().parent.parent

    return root_folder


def _import_local_file(path: Path, *, project_root: Path) -> None:
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.parent.resolve()

    module_name = ".".join(
        path.absolute().relative_to(project_root.absolute()).with_suffix("").parts
    )
    # logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)
