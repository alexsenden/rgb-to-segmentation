# RGB to Segmentation

A Python package for processing and cleaning segmentation images. This package provides tools to convert RGB images to segmentation masks using palette-based color mapping and neural network-based refinement.

## Features

- **Palette-based Cleaning**: Clean noisy segmentation images by mapping pixels to the nearest colors in a predefined palette, with optional morphological operations to refine boundaries.
- **Neural Network Refinement**: Use a trained pixelwise classifier to refine segmentation masks using PyTorch Lightning.
- **Command-Line Interface**: Unified CLI for cleaning with method selection, plus separate training command.
- **Programmatic API**: Direct access to cleaning and training functions for integration into other workflows.

## Installation

Install from PyPI:

```bash
pip install rgb-to-segmentation
```

Or install from source:

```bash
git clone https://github.com/alexsenden/rgb-to-segmentation.git
cd rgb-to-segmentation
pip install .
```

## Usage

### Cleaning Noisy Segmentation Images

Use the `segment-clean` command to clean segmentation images using various methods:

#### Palette-based cleaning:

```bash
segment-clean --method palette --input_dir /path/to/input --output_dir /path/to/output --colour_map "0,0,0;255,0,0;0,255,0" --output_type rgb
```

#### Neural network-based cleaning:

```bash
segment-clean --method nn --input_dir /path/to/input --output_dir /path/to/output --model_path /path/to/model.ckpt --colour_map "0,0,0;255,0,0;0,255,0" --output_type index
```

You can also provide colours via file with `--colour_map_file /path/to/colours.txt` (one `r,g,b` per line). The CLI parses colours and constructs the palette/colour map internally, mirroring the Python API which accepts parsed structures (NumPy array for palette, dictionary for colour map).

Options:

- `--method`: Cleaning method ('palette' or 'nn')
- `--input_dir`: Path to input directory containing images
- `--output_dir`: Directory where cleaned images will be written
- `--inplace`: Overwrite input images in place
- `--exts`: Comma-separated list of allowed image extensions
- `--name_filter`: Only process files whose name contains this substring
- `--output_type`: Output format ('rgb' or 'index')

For palette method:
- `--colour_map`: Semicolon-separated list of RGB triples
- `--colour_map_file`: Path to a file listing RGB triples
- `--morph_kernel_size`: Size of morphological kernel for boundary cleaning

For nn method:
- `--model_path`: Path to trained model file
- `--colour_map`: Semicolon-separated list of RGB triples
- `--colour_map_file`: Path to a file listing RGB triples

### Training the Neural Network Model

Train a pixelwise classifier to refine segmentation masks:

```bash
segment-train --image_dir /path/to/noisy_images --label_dir /path/to/labels --output_dir /path/to/model_output --colour_map "0,0,0;255,0,0;0,255,0"
```

Options:
- `--image_dir`: Path to directory containing noisy images
- `--label_dir`: Path to directory containing target RGB labels
- `--output_dir`: Directory where model weights will be saved
- `--colour_map`: Semicolon-separated list of RGB triples
- `--colour_map_file`: Path to a file listing RGB triples
- `--model_type`: The type of model to train (default: pixelwise)

Note that one label image may have multiple corresponding noisy masks. Labels are matched to noisy masks whose filenames contain the label file basename (pre-extension name, i.e. `my_image.png` -> `my_image`).

## API

You can also use the package programmatically:

```python
import numpy as np
from rgb_to_segmentation import clean, nn, train, utils

# Palette cleaning
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
palette = np.asarray(colours, dtype=np.uint8)
clean.clean_segmentation(input_dir="/path/to/input", output_dir="/path/to/output", palette=palette, output_type="index")

# NN inference
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
colour_map = {i: rgb for i, rgb in enumerate(colours)}
nn.run_inference(input_dir="/path/to/input", output_dir="/path/to/output", model_path="/path/to/model.ckpt", colour_map=colour_map, output_type="rgb")

# Train model
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
colour_map = {i: rgb for i, rgb in enumerate(colours)}
train.train_model(image_dir="/path/to/images", label_dir="/path/to/labels", output_dir="/path/to/output", colour_map=colour_map)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
