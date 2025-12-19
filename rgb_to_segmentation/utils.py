from typing import List, Tuple


def parse_colours_from_string(colours_str: str) -> List[Tuple[int, int, int]]:
    """Parse a semicolon-separated string of RGB triples into a list of tuples."""

    parts = [p.strip() for p in colours_str.split(";") if p.strip()]
    colours = []

    for p in parts:
        rgb = tuple(int(x) for x in p.split(","))

        if len(rgb) != 3:
            raise ValueError(f"Invalid colour triple: {p}")
        colours.append(rgb)

    return colours


def parse_colours_from_file(path: str) -> List[Tuple[int, int, int]]:
    """Parse a file with one RGB triple per line into a list of tuples."""

    colours = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            rgb = tuple(int(x) for x in line.split(","))
            if len(rgb) != 3:
                raise ValueError(f"Invalid colour triple in file {path}: {line}")

            colours.append(rgb)

    if not colours:
        raise ValueError(f"No colours found in file: {path}")

    return colours
