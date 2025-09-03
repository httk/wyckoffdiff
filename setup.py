from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wyckoff-gen",
    packages=["wyckoff_generation"],
    version="0.1.0",
    description="Generative modeling of Wyckoff representations of materials",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch==2.3.0",
        "torch_geometric==2.5.3",
        "tqdm",
        "numpy==1.26.4",
        'Aviary @ git+https://github.com/CompRhys/aviary@v1.1.1',
        "matplotlib",
        "scikit-learn",
        "pandas",
        "wandb",
        "pre-commit",
        "pyxtal",
        "mace-torch",
    ],
)
