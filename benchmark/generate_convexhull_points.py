import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate uniformly random points for convex hull benchmarks."
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
        "--low",
        type=float,
        default=-10000000000,
        help="Lower bound of the uniform distribution. Defaults to 0.0.",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=10000000000.0,
        help="Upper bound of the uniform distribution. Defaults to 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output.",
    )
    return parser.parse_args()


def write_points(path, rng, n_points, dimension, low, high, chunk_size, write_header):
    remaining = n_points

    with open(path, "w", encoding="utf-8") as file:
        if write_header:
            file.write(f"{n_points} {dimension}\n")

        while remaining > 0:
            count = min(remaining, chunk_size)
            points = rng.uniform(low, high, size=(count, dimension))
            np.savetxt(file, points, fmt="%.18e")
            remaining -= count


def main():
    args = parse_args()

    if args.dimension <= 0:
        raise ValueError("dimension must be positive.")
    if args.low >= args.high:
        raise ValueError("low must be smaller than high.")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")
    if any(size <= 0 for size in args.sizes):
        raise ValueError("all sizes must be positive.")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for n_points in args.sizes:
        path = os.path.join(args.output_dir, f"data/{args.prefix}_{n_points}_d{args.dimension}.txt")
        write_points(
            path,
            rng,
            n_points,
            args.dimension,
            args.low,
            args.high,
            args.chunk_size,
            args.header,
        )
        print(path)


if __name__ == "__main__":
    main()
