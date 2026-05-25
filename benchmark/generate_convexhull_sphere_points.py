import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random points on or around a sphere/hypersphere for convex hull benchmarks."
    )
    parser.add_argument("dimension", type=int, help="Dimension of each point.")
    parser.add_argument(
        "-d",
        "--output-dir",
        default=".",
        help="Directory where generated files are written. Defaults to current directory.",
    )
    parser.add_argument(
        "--prefix",
        default="points",
        help="Output filename prefix. Defaults to points.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            10,
            100,
            1000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            20_000_000,
            30_000_000,
            40_000_000,
            45_000_000,
            50_000_000,
        ],
        help="Point counts to generate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of points generated per write chunk. Defaults to 1000000.",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        default=True,
        help="Write 'n_points dimension' as the first line.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1_000_000,
        help="Sphere radius. Defaults to 1000000.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.0,
        help=(
            "Half-width of the radial shell around the radius. "
            "Defaults to 0.0, which generates points on the sphere surface."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output.",
    )
    return parser.parse_args()


def generate_sphere_points(rng, count, dimension, radius, thickness):
    directions = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)

    while np.any(norms == 0):
        zero_norms = norms[:, 0] == 0
        directions[zero_norms] = rng.normal(
            size=(np.count_nonzero(zero_norms), dimension)
        )
        norms[zero_norms] = np.linalg.norm(
            directions[zero_norms],
            axis=1,
            keepdims=True,
        )

    directions /= norms

    if thickness == 0:
        return radius * directions

    inner_radius = radius - thickness
    outer_radius = radius + thickness
    radii = rng.uniform(
        inner_radius**dimension,
        outer_radius**dimension,
        size=(count, 1),
    ) ** (1.0 / dimension)

    return radii * directions


def write_points(
    path,
    rng,
    n_points,
    dimension,
    radius,
    thickness,
    chunk_size,
    write_header,
):
    remaining = n_points

    with open(path, "w", encoding="utf-8") as file:
        if write_header:
            file.write(f"{n_points} {dimension}\n")

        while remaining > 0:
            count = min(remaining, chunk_size)
            points = generate_sphere_points(
                rng,
                count,
                dimension,
                radius,
                thickness,
            )
            np.savetxt(file, points, fmt="%.0f")
            remaining -= count


def main():
    args = parse_args()

    if args.dimension <= 0:
        raise ValueError("dimension must be positive.")
    if args.radius <= 0:
        raise ValueError("radius must be positive.")
    if args.thickness < 0:
        raise ValueError("thickness must be non-negative.")
    if args.thickness >= args.radius:
        raise ValueError("thickness must be smaller than radius.")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")
    if any(size <= 0 for size in args.sizes):
        raise ValueError("all sizes must be positive.")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)

    for n_points in args.sizes:
        path = os.path.join(
            args.output_dir,
            "data",
            f"sphere_{args.prefix}_{n_points}_d{args.dimension}.txt",
        )
        write_points(
            path,
            rng,
            n_points,
            args.dimension,
            args.radius,
            args.thickness,
            args.chunk_size,
            args.header,
        )
        print(path)


if __name__ == "__main__":
    main()
