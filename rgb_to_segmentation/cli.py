import argparse

from . import clean, nn, train


def main_clean():
    parser = argparse.ArgumentParser(
        description="Clean segmentation images using various methods."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["palette", "nn"],
        help="Cleaning method to use: 'palette' for color palette mapping, 'nn' for neural network.",
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
    # Palette-specific args
    parser.add_argument(
        "--colours",
        type=str,
        help="Semicolon-separated list of RGB triples for palette method.",
    )
    parser.add_argument(
        "--colours_file",
        type=str,
        help="Path to a file listing RGB triples for palette method.",
    )
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
    parser.add_argument(
        "--colour_map",
        type=str,
        help="Semicolon-separated list of RGB triples for nn method.",
    )
    parser.add_argument(
        "--colour_map_file",
        type=str,
        help="Path to a file listing RGB triples for nn method.",
    )

    args = parser.parse_args()

    if args.method == "palette":
        if not args.colours and not args.colours_file:
            parser.error("--colours or --colours_file required for palette method")
        clean.clean_segmentation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            inplace=args.inplace,
            colours=args.colours,
            colours_file=args.colours_file,
            exts=args.exts,
            name_filter=args.name_filter,
            morph_kernel_size=args.morph_kernel_size,
        )
    elif args.method == "nn":
        if not args.model_path:
            parser.error("--model_path required for nn method")
        if not args.colour_map and not args.colour_map_file:
            parser.error("--colour_map or --colour_map_file required for nn method")
        colour_map = train.get_colour_map(args.colour_map, args.colour_map_file)
        nn.run_inference(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            inplace=args.inplace,
            model_path=args.model_path,
            colour_map=colour_map,
            exts=args.exts,
            name_filter=args.name_filter,
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
        default="pixelwise",
        help="The type of model to train.",
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

    train.train_model(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        colour_map_str=args.colour_map,
        colour_map_file=args.colour_map_file,
        model_type=args.model_type,
    )
