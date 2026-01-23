# RGB to Segmentation

A Python package for processing and cleaning segmentation images. This package provides tools to convert RGB images to segmentation masks using palette-based color mapping and neural network-based refinement.

## Features

- **Palette-based Cleaning**: Clean noisy segmentation images by mapping pixels to the nearest colors in a predefined palette, with optional morphological operations to refine boundaries.
- **Strict Palette Mapping**: Directly map RGB values to class indices with strict validation - throws an error if any pixel value is not in the colour map.
- **Neural Network Refinement**: Use trained neural network models to refine segmentation masks using PyTorch Lightning:
  - **Pixelwise Classifier**: MLP-based pixel-by-pixel classification
  - **CNN Decoder**: Convolutional encoder-decoder architecture for spatial context
- **Flexible Input Types**: Accept both NumPy arrays and PyTorch tensors, with output type matching input type.
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
# Works with both pixelwise and CNN decoder models
segment-clean --method nn --input_dir /path/to/input --output_dir /path/to/output --model_path /path/to/model.ckpt --colour_map "0,0,0;255,0,0;0,255,0" --output_type index
```

#### Strict palette mapping:

```bash
segment-clean --method strict_palette --input_dir /path/to/input --output_dir /path/to/output --colour_map "0,0,0;255,0,0;0,255,0" --output_type index
```

You can also provide colours via file with `--colour_map_file /path/to/colours.txt` (one `r,g,b` per line). The CLI parses colours and constructs the palette/colour map internally, mirroring the Python API which accepts parsed structures (NumPy array for palette, dictionary for colour map).

Options:

- `--method`: Cleaning method ('palette', 'nn', or 'strict_palette')
- `--input_dir`: Path to input directory containing images
- `--output_dir`: Directory where cleaned images will be written
- `--inplace`: Overwrite input images in place
- `--exts`: Comma-separated list of allowed image extensions
- `--name_filter`: Only process files whose name contains this substring
- `--output_type`: Output format ('rgb' or 'index')
- `--colour_map`: Semicolon-separated list of RGB triples
- `--colour_map_file`: Path to a file listing RGB triples

For palette method:

- `--morph_kernel_size`: Size of morphological kernel for boundary cleaning

For nn method (both pixelwise and CNN decoder models):

- `--model_path`: Path to trained model file

The strict_palette method requires no additional options beyond the common ones.

### Training the Neural Network Model

Train a neural network model to refine segmentation masks. Choose between pixelwise MLP or CNN decoder:

```bash
# Train pixelwise classifier (default)
segment-train --image_dir /path/to/noisy_images --label_dir /path/to/labels --output_dir /path/to/model_output --colour_map "0,0,0;255,0,0;0,255,0"

# Train CNN decoder
segment-train --image_dir /path/to/noisy_images --label_dir /path/to/labels --output_dir /path/to/model_output --model_type cnn_decoder --colour_map "0,0,0;255,0,0;0,255,0"
```

Options:

- `--image_dir`: Path to directory containing noisy images
- `--label_dir`: Path to directory containing target RGB labels
- `--output_dir`: Directory where model weights will be saved
- `--colour_map`: Semicolon-separated list of RGB triples
- `--colour_map_file`: Path to a file listing RGB triples
- `--model_type`: The type of model to train ('pixel_decoder' or 'cnn_decoder', default: pixel_decoder)

Note that one label image may have multiple corresponding noisy masks. Labels are matched to noisy masks whose filenames contain the label file basename (pre-extension name, i.e. `my_image.png` -> `my_image`).

## API

You can also use the package programmatically:

```python
import numpy as np
from rgb_to_segmentation import clean, nn, train, utils, clean_image

# Palette cleaning
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
palette = np.asarray(colours, dtype=np.uint8)
clean.clean_segmentation(input_dir="/path/to/input", output_dir="/path/to/output", palette=palette, output_type="index")

# NN inference (works with both pixelwise and CNN decoder models)
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
colour_map = {i: rgb for i, rgb in enumerate(colours)}
nn.run_inference(input_dir="/path/to/input", output_dir="/path/to/output", model_path="/path/to/model.ckpt", colour_map=colour_map, output_type="rgb")

# Train pixelwise model
colours = utils.parse_colours_from_string("0,0,0;255,0,0;0,255,0")
colour_map = {i: rgb for i, rgb in enumerate(colours)}
train.train_model(image_dir="/path/to/images", label_dir="/path/to/labels", output_dir="/path/to/output", colour_map=colour_map)

# Train CNN decoder model
train.train_model(image_dir="/path/to/images", label_dir="/path/to/labels", output_dir="/path/to/output", colour_map=colour_map, model_type="cnn_decoder")

# Single-image cleaning (programmatic-only API)
# Accepts both numpy arrays and torch tensors; output type matches input type

# Palette method (returns RGB)
import numpy as np
rgb_out = clean_image(
	image_array=np.zeros((512, 512, 3), dtype=np.uint8),
	method="palette",
	colour_map=colour_map,
	morph_kernel_size=3,
	output_type="rgb",
)

# Strict palette method (returns index mask, validates all pixels are in colour_map)
index_out = clean_image(
	image_array=np.zeros((512, 512, 3), dtype=np.uint8),
	method="strict_palette",
	colour_map=colour_map,
	output_type="index",
)

# Pixel decoder method (returns index mask, works with pixelwise or CNN models)
index_out = clean_image(
	image_array=np.zeros((512, 512, 3), dtype=np.uint8),
	method="pixel_decoder",
	model=None,  # Provide a loaded model instance
	colour_map=colour_map,
	output_type="index",
)

# CNN decoder method (returns index mask)
index_out = clean_image(
	image_array=np.zeros((512, 512, 3), dtype=np.uint8),
	method="cnn_decoder",
	model=None,  # Provide a loaded CNN decoder model instance
	colour_map=colour_map,
	output_type="index",
)

# Using PyTorch tensors
import torch
tensor_input = torch.zeros((512, 512, 3), dtype=torch.uint8)
tensor_out = clean_image(
	image_array=tensor_input,
	method="strict_palette",
	colour_map=colour_map,
	output_type="rgb",
)
# tensor_out will be a torch.Tensor with same dtype and device as tensor_input
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
