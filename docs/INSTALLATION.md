# Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/AshrithSagar/Classification-Fungal10x.git
    cd Classification-Fungal10x
    ```

2. Optionally, create a virtual environment and activate it.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    Or use `conda` to create a virtual environment.

    ```bash
    conda create --name clfx python=3.9
    conda activate clfx
    ```

3. Install the required packages.

    ```bash
    pip install -r requirements.txt
    ```

## Additional loss functions

Used by model CLAM.

(Preferably install it somewhere outside the project directory.)

```bash
git clone https://github.com/oval-group/smooth-topk.git
pip install smooth-topk/
```
