from utility import generate_linear_points
from query_comparison import run_query_comparison


def main(volume_name=None):
    run_query_comparison(
        point_generator=lambda D, H, W, n: generate_linear_points((D, H, W)),
        title="Grid representation comparison (grid traversal queries)",
        file_suffix="_linear",
        volume_name=volume_name,
    )


if __name__ == "__main__":
    main()
