import argparse

import numpy as np

from . import clean, nn, train, utils


def main_clean():
    parser = argparse.ArgumentParser(
        description="Clean segmentation images using various methods."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["palette", "nn"],
        help="Cleaning method to use: 'palette' for color palette mapping, 'nn' for neural network (pixelwise or CNN).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory where cleaned images will be written. Required if not inplace.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input images in place.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.tiff,.bmp,.gif",
        help="Comma-separated list of allowed image extensions.",
    )
    parser.add_argument(
        "--name_filter",
        type=str,
        default="",
        help="Only process files whose name contains this substring.",
    )

    parser.add_argument(
        "--output_type",
        type=str,
        choices=["rgb", "index"],
        default="rgb",
        help="Output format: 'rgb' colour image or 'index' mask.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--colour_map",
        type=str,
        help="Semicolon-separated list of RGB triples.",
    )
    group.add_argument(
        "--colour_map_file",
        type=str,
        help="Path to a file listing RGB triples.",
    )

    # Palette-specific args
    parser.add_argument(
        "--morph_kernel_size",
        type=int,
        default=3,
        help="Size of morphological kernel for palette method.",
    )

    # NN-specific args
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model for nn method.",
    )

    args = parser.parse_args()

    if args.colour_map_file:
        colours = utils.parse_colours_from_file(args.colour_map_file)
    else:
        colours = utils.parse_colours_from_string(args.colour_map)

    if args.method == "palette":
        palette = np.asarray(colours, dtype=np.uint8)

        clean.clean_segmentation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            inplace=args.inplace,
            palette=palette,
            exts=args.exts,
            name_filter=args.name_filter,
            morph_kernel_size=args.morph_kernel_size,
            output_type=args.output_type,
        )

    elif args.method == "nn":
        if not args.model_path:
            parser.error("--model_path required for nn method")

        colour_map = {i: rgb for i, rgb in enumerate(colours)}
        nn.run_inference(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            inplace=args.inplace,
            model_path=args.model_path,
            colour_map=colour_map,
            exts=args.exts,
            name_filter=args.name_filter,
            output_type=args.output_type,
        )


def main_train():
    parser = argparse.ArgumentParser(
        description="Train a neural network model for segmentation cleaning."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to directory containing noisy images.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Path to directory containing target RGB labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model weights will be saved.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pixelwise", "cnn_decoder"],
        default="pixelwise",
        help="The type of model to train: 'pixelwise' for MLP or 'cnn_decoder' for CNN-based decoder.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--colour_map",
        type=str,
        help="Semicolon-separated list of RGB triples.",
    )
    group.add_argument(
        "--colour_map_file",
        type=str,
        help="Path to a file listing RGB triples.",
    )

    args = parser.parse_args()

    if args.colour_map_file:
        colours = utils.parse_colours_from_file(args.colour_map_file)
    else:
        colours = utils.parse_colours_from_string(args.colour_map)
    colour_map = {i: rgb for i, rgb in enumerate(colours)}

    train.train_model(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        colour_map=colour_map,
        model_type=args.model_type,
    )
