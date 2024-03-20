# Classification-Fungal10x

![GitHub](https://img.shields.io/github/license/AshrithSagar/Classification-Fungal10x) ![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/Classification-Fungal10x) [![CodeFactor](https://www.codefactor.io/repository/github/AshrithSagar/Classification-Fungal10x/badge)](https://www.codefactor.io/repository/github/AshrithSagar/Classification-Fungal10x)

Fungal classification | Multiple Instance Learning

## Installation

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

## Usage

### Config file

The configuration file contains various settings and parameters that control the behavior and settings of the project.
Refer to the [config-template.yaml](config-template.yaml) file for more information.

### Directory structure

The directory structure of the created datasets is as follows:

Slide directory structure:

```markdown
dataset
└── <ds-name>-slides
    ├── train
    │   ├── fungal
    │   │   ├── F-slide-1.png
    │   │   ├── F-slide-2.png
    │   │   └── ...
    │   └── non-fungal
    │       ├── NF-slide-1.png
    │       ├── NF-slide-2.png
    │       └── ...
    └── test
        ├── fungal
        │   ├── F-slide-1.png
        │   ├── F-slide-2.png
        │   └── ...
        └── non-fungal
            ├── NF-slide-1.png
            ├── NF-slide-2.png
            └── ...
```

Patches directory structure:

```markdown
dataset
└── <ds-name>-<downsample_dimensions>
    ├── fold-0
    │   ├── train
    │   │   ├── fungal
    │   │   │   ├── F-patch-1.png
    │   │   │   ├── F-patch-2.png
    │   │   │   └── ...
    │   │   └── non-fungal
    │   │       ├── NF-patch-1.png
    │   │       ├── NF-patch-2.png
    │   │       └── ...
    │   ├── val
    │   │   ├── fungal
    │   │   │   ├── F-patch-1.png
    │   │   │   ├── F-patch-2.png
    │   │   │   └── ...
    │   │   └── non-fungal
    │   │       ├── NF-patch-1.png
    │   │       ├── NF-patch-2.png
    │   │       └── ...
    │   └── test
    │       ├── fungal
    │       │   ├── F-patch-1.png
    │       │   ├── F-patch-2.png
    │       │   └── ...
    │       └── non-fungal
    │           ├── NF-patch-1.png
    │           ├── NF-patch-2.png
    │           └── ...
    ├── fold-1
    │   └── ...
    └── ...
```

## License

This project falls under the [MIT License](LICENSE).
