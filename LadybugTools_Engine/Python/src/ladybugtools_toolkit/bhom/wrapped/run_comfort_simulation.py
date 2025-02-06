import argparse
import matplotlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_path",
        help="path to weather file to use in simulation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-in",
        "--json_args",
        help="JSON file input containing necessary information to run the simulation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ws",
        "--wind_speed_multiplier",
        help="wind speed multiplier to apply to epw file before simulation.",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-r",
        "--return_file",
        help="json file to write return data to.",
        type=str,
        required=True,
        )

    args = parser.parse_args()
    matplotlib.use("Agg")
    comfort_simulation(args.epw_path, args.json_args, args.return_file, args.wind_speed_multiplier)