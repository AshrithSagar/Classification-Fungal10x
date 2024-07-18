# Directory structure

The directory structure of the created datasets is as follows:

## Slide directory structure

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

## Patches directory structure

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

## MIL directory structure

```markdown
dataset
└── <ds-name>-MIL-<downsample_dimensions>
    ├── train
    │   ├── F-slide-1
    │   │   ├── F-patch-1.png
    │   │   ├── F-patch-2.png
    │   │   └── ...
    │   ├── ...
    │   ├── NF-slide-1
    │   │   ├── NF-patch-1.png
    │   │   ├── NF-patch-2.png
    │   │   └── ...
    │   └── ...
    └── test
        ├── F-slide-1
        │   ├── F-patch-1.png
        │   ├── F-patch-2.png
        │   └── ...
        ├── ...
        ├── NF-slide-1
        │   ├── NF-patch-1.png
        │   ├── NF-patch-2.png
        │   └── ...
        └── ...
```
