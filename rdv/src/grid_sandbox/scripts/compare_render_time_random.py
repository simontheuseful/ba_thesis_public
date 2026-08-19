from utility import generate_random_points
from query_comparison import run_query_comparison


def main(volume_name=None):
    run_query_comparison(
        point_generator=lambda D, H, W, n: generate_random_points(n),
        title="Grid representation comparison (random point queries)",
        file_suffix="_random",
        volume_name=volume_name,
    )


if __name__ == "__main__":
    main()
