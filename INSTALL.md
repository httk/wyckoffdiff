# Installation

Perform these steps to install WyckoffDiff and its dependencies.

1. Clone the WyckoffDiff repository:
    ```
    git clone https://github.com/httk/wyckoffdiff.git
    cd wyckoffdiff
    ```

1. Create a conda environment called ```wyckoffdiff``` with Python 3.12 and activate it using
    ```
    conda create -n wyckoffdiff python=3.12
    conda activate wyckoffdiff
    ```

    **Alternatively**, if your system Python version is recent enough, you can instead use a Python venv:
    ```
    mkdir -p data/deps
    python3 -m venv data/deps/venv
    source data/deps/venv/bin/activate
    ```

1. Install ```wyckoff_generation``` as a package with its dependencies by
    ```
    pip install -e .
    ```

1. Install pre-commit hooks by
    ```
    pre-commit install
    ```

At this point you are ready to continue with the usage instructions in README.md.

(To activate the environment created above in a new shell, just do: `conda activate wyckoffdiff` or `source data/deps/venv/bin/activate`.)
