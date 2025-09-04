# This file is based on code made available under the MIT license by FAIR Chemistry available from their repository as https://github.com/facebookresearch/fairchem/blob/dbaefaed40eee2844d033c78ccd2fe68976ebcb6/ocpmodels/common/registry.py
#
# Modifications Copyright (c) 2025 The High-Throughput Toolkit
#
# The original copyright header from the fairchem file follows below:
#
# MIT License
#
# Copyright (c) Meta, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.

Import the global registry object using

``from wyckoff_generation.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a model: ``@registry.register_model``
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, ClassVar, TypeVar, Union

R = TypeVar("R")
NestedDict = dict[str, Union[str, Callable[..., Any], "NestedDict"]]


def _get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `wyckoff_generation.runners.base_runner.BaseRunner`
    # we can use importlib to get the module (e.g., `wyckoff_generation.runners.base_runner`)
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"Could not import module `{module_name}` for import `{name}`"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class `{class_name}` from module `{module_name}`"
        ) from e


class Registry:
    r"""Class for registry object which acts as central source of truth."""

    mapping: ClassVar[NestedDict] = {
        # Mappings to respective classes.
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "logger_name_mapping": {},
        "runner_name_mapping": {},
        "d3pm_transition_name_mapping": {},
        "gnn_name_mapping": {},
        "datainfo_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_runner(cls, name: str):
        r"""Register a new runner to registry with key 'name'
        Args:
            name: Key with which the runner will be registered.
        Usage::
            from wyckoff_generation.common.registry import registry
            from wyckoff_generation.runners import BaseRunner
            @registry.register_task("train_gen_model")
            class GenerativeModelTrainer(BaseRunner):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["runner_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_logger(cls, name: str):
        r"""Register a logger to registry with key 'name'

        Args:
            name: Key with which the logger will be registered.

        Usage:
            from wyckoff_generation.common.registry import registry

            @registry.register_logger("wandb")
            class WandBLogger(BaseLogger):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            from wyckoff_generation.common.logging import BaseLogger

            assert issubclass(
                func, BaseLogger
            ), "All loggers must inherit BaseLogger class"
            cls.mapping["logger_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls, name: str):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage::

            from wyckoff_generation.common.registry import registry
            from wyckoff_generation.dataset.dataset import WyckoffDataset

            @registry.register_dataset("base")
            class WyckoffDatset(InMemoryDataset):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            from wyckoff_generation.datasets.dataset import WyckoffDataset

            assert issubclass(
                func, WyckoffDataset
            ), "All datasets must inherit BaseDataset class"
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_datainfo(cls, name: str):
        r"""Register a dataset info to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage::

            from wyckoff_generation.common.registry import registry

            @registry.register_datainfo("base")
            class WBMDatasetInfo:
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:

            cls.mapping["datainfo_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name: str):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.
        Usage:

            import torch.nn as nn

            @registry.register_model("d3pm")
            class D3PM(nn.Module):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_gnn(cls, name: str):
        r"""Register a GNN to registry with key 'name'

        Args:
            name: Key with which the GNN will be registered.
        Usage:

            import torch.nn as nn

            @registry.register_gnn("base")
            class WyckoffGNN(nn.Module):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["gnn_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_d3pm_transition(cls, name: str):
        r"""Register a D3PM transition to registry with key 'name'

        Args:
            name: Key with which the model will be registered.
        Usage:

            import torch.nn as nn

            @registry.register_d3pm_transition("uniform")
            class D3PMUniformTransition(nn.Module):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["d3pm_transition_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name: str, obj) -> None:
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from wyckoff_generation.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def __import_error(cls, name: str, mapping_name: str) -> RuntimeError:
        kind = mapping_name[: -len("_name_mapping")]
        mapping = cls.mapping.get(mapping_name, {})
        existing_keys: list[str] = list(mapping.keys())

        if len(existing_keys) == 0:
            raise RuntimeError(
                f"Registry for {mapping_name} is empty. You may have forgot to load a module."
            )

        existing_cls_path = (
            mapping.get(existing_keys[-1], None) if existing_keys else None
        )
        if existing_cls_path is not None:
            existing_cls_path = (
                f"{existing_cls_path.__module__}.{existing_cls_path.__qualname__}"
            )
        else:
            existing_cls_path = "wyckoff_generation.runners.GenModelTrainer"

        existing_keys = [f"'{name}'" for name in existing_keys]
        existing_keys = ", ".join(existing_keys[:-1]) + " or " + existing_keys[-1]
        existing_keys_str = f" (one of {existing_keys})" if existing_keys else ""

        if name is None:
            first_sentence = (
                f"Seems like {kind} is None, probably because you have not set it. "
            )
        else:
            first_sentence = f"Failed to find the {kind} '{name}'. "
        return RuntimeError(
            first_sentence
            + f"You may either use a {kind} from the registry{existing_keys_str} "
            f"or provide the full import path to the {kind} (e.g., '{existing_cls_path}')."
        )

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        if name is None:
            raise cls.__import_error(name, mapping_name)

        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}` (e.g., `wyckoff_generation.runners.GenModelTrainer`)
        if name.count(".") < 1:
            raise cls.__import_error(name, mapping_name)

        try:
            return _get_absolute_mapping(name)
        except RuntimeError as e:
            raise cls.__import_error(name, mapping_name) from e

    @classmethod
    def get_dataset_class(cls, name: str):
        return cls.get_class(name, "dataset_name_mapping")

    @classmethod
    def get_datainfo_class(cls, name: str):
        return cls.get_class(name, "datainfo_name_mapping")

    @classmethod
    def get_logger_class(cls, name: str):
        return cls.get_class(name, "logger_name_mapping")

    @classmethod
    def get_runner_class(cls, name: str):
        return cls.get_class(name, "runner_name_mapping")

    @classmethod
    def get_model_class(cls, name: str):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_d3pm_transition_class(cls, name: str):
        return cls.get_class(name, "d3pm_transition_name_mapping")

    @classmethod
    def get_gnn_class(cls, name: str):
        return cls.get_class(name, "gnn_name_mapping")

    @classmethod
    def get(cls, name: str, default=None, no_warning: bool = False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from wyckoff_generation.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        split_name = name.split(".")
        value = cls.mapping["state"]
        for subname in split_name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                f"Key {original_name} is not present in registry, returning default value "
                f"of {default}"
            )
        return value

    @classmethod
    def unregister(cls, name: str):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from wyckoff_generation.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
