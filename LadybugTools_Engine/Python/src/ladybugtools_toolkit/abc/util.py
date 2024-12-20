"""Utility functions to enable use of the ABC model API."""

import csv
import json
from pathlib import Path

import requests

from ..bhom.logging import CONSOLE_LOGGER


def read_json_file(file_path: str) -> str:
    """Read a JSON file and return its content as a string

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        str: The content of the JSON file.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        CONSOLE_LOGGER.error("Error: File not found - %s", file_path)
        raise
    except IOError as e:
        CONSOLE_LOGGER.error("Error: Unable to read file %s - %s", file_path, e)
        raise


def post_json_data(url: str, data: str, headers: dict, timeout: int) -> requests.Response:
    """Post JSON data to a URL and return the response.

    Args:
        url (str): The URL to post the JSON data to.
        data (str): The JSON data to post.
        headers (dict): The headers to include in the request.
        timeout (int): Time in seconds to wait for the server to send data.

    Returns:
        requests.Response: The response from the server.
    """

    try:
        return requests.post(url, data=data, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        CONSOLE_LOGGER.error("Error: The request timed out after %s seconds.", timeout)
        raise
    except requests.exceptions.RequestException as e:
        CONSOLE_LOGGER.error("Error: %s", e)
        raise


def save_output(file_path: str, data: str) -> None:
    """Save the output data to a file.

    Args:
        file_path (str): The path to the file to save the output data to.
        data (str): The output data to save.
    """

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(data)
        CONSOLE_LOGGER.info("Output has been saved to %s", file_path)
    except IOError as e:
        CONSOLE_LOGGER.error("Error: Unable to write to file %s - %s", file_path, e)
        raise


def call_abc_model_api(input_file: Path, output_file: Path, url: str, timeout: int = 10) -> None:
    """Pass the input file (JSON) to the ABC model API and save the output to
    the output file (JSON).

    Args:
        input_file (Path):
            A JSON file containing the input data for the ABC model.
        output_file (Path):
            A JSON file to save the output data from the ABC model simulation.
        url (str):
            The URL of the ABC model API.
        timeout (int, optional):
            Time in seconds to wait for the server to send data. Defaults to 10.
    """

    CONSOLE_LOGGER.info("Passing JSON to API")

    input_text = read_json_file(input_file)
    headers = {"Content-Type": "application/json"}
    response = post_json_data(url, input_text, headers, timeout)
    if response.status_code == 200:
        save_output(output_file, response.text)
    else:
        CONSOLE_LOGGER.error("Response error: %s", response.status_code)
        CONSOLE_LOGGER.error(response.text)
        response.raise_for_status()


def output_json_to_csv(output_json_file_path: str, csv_file_path: str) -> None:
    """Convert the output JSON file from the ABC model API to a CSV file.

    Args:
        output_json_file_path (str): The path to the output JSON file.
        csv_file_path (str): The path to the CSV file to write the data to.
    """

    with open(output_json_file_path, encoding="utf-8") as json_file:
        data = json.load(json_file)

    rows = []

    for result in data["results"]:
        base_data = {"elapsed_time": result["time"], "Met": result["met"], "Clo": result["clo"]}

        overall_data = {
            "Ta": result["overall"]["ta"],
            "MRT": result["overall"]["mrt"],
            "RH": result["overall"]["rh"],
            "Velocity": result["overall"]["v"],
            "Solar": result["overall"]["solar"],
            "Overall_Comfort": result["overall"]["comfort"],
            "Overall_Comfort_weighted": result["overall"]["comfort_weighted"],
            "Overall_Sensation": result["overall"]["sensation"],
            "Overall_Sensation_Linear": result["overall"]["sensation_linear"],
            "Overall_Sensation_Weighted": result["overall"]["sensation_weighted"],
            "MeanSkinTemp": result["overall"]["tskin"],
            "Tblood": result["overall"]["tblood"],
            "Tneutral": result["overall"]["tneutral"],
            "PMV": result["overall"]["pmv"],
            "PPD": result["overall"]["ppd"],
            "EHT": result["overall"]["eht"],
            "Qmet": result["overall"]["q_met"],
            "Qconv": result["overall"]["q_conv"],
            "Qrad": result["overall"]["q_rad"],
            "Qsolar": result["overall"]["q_solar"],
            "Qresp": result["overall"]["q_resp"],
            "Qsweat": result["overall"]["q_sweat"],
        }
        base_data.update(overall_data)

        for segment, segment_data in result["segments"].items():
            segment_data_prefixed = {
                f"Tskin-{segment}": segment_data["tskin"],
                f"Tcore-{segment}": segment_data["tcore"],
                f"Sens-{segment}": segment_data["sensation"],
                f"Sens_weighted-{segment}": segment_data["sensation_weighted"],
                f"Comfort-{segment}": segment_data["comfort"],
                f"Comfort_weighted-{segment}": segment_data["comfort_weighted"],
                f"EHT-{segment}": segment_data["eht"],
                f"Tskin_set-{segment}": segment_data["tskin_set"],
                f"Tskin_set_reg-{segment}": segment_data["tskin_set_reg"],
            }
            base_data.update(segment_data_prefixed)

        rows.append(base_data)

    # Define the desired column order
    column_order = [
        "elapsed_time",
        "Ta",
        "MRT",
        "RH",
        "Velocity",
        "Solar",
        "Clo",
        "Met",
        "Overall_Comfort",
        "Overall_Comfort_weighted",
        "Overall_Sensation",
        "Overall_Sensation_Linear",
        "Overall_Sensation_Weighted",
        "MeanSkinTemp",
        "Tblood",
        "Tneutral",
        "PMV",
        "PPD",
        "EHT",
        "Qmet",
        "Qconv",
        "Qrad",
        "Qsolar",
        "Qresp",
        "Qsweat",
    ]

    # Add segment columns to the order
    segments = [
        "Head",
        "Chest",
        "Back",
        "Pelvis",
        "Left Upper Arm",
        "Right Upper Arm",
        "Left Lower Arm",
        "Right Lower Arm",
        "Left Hand",
        "Right Hand",
        "Left Thigh",
        "Right Thigh",
        "Left Lower Leg",
        "Right Lower Leg",
        "Left Foot",
        "Right Foot",
    ]

    for segment in segments:
        column_order.extend(
            [
                f"Tskin-{segment}",
                f"Tcore-{segment}",
                f"Sens-{segment}",
                f"Sens_weighted-{segment}",
                f"Comfort-{segment}",
                f"Comfort_weighted-{segment}",
                f"EHT-{segment}",
                f"Tskin_set-{segment}",
                f"Tskin_set_reg-{segment}",
            ]
        )

    # Ensure all columns are present in each row
    for row in rows:
        for col in column_order:
            if col not in row:
                row[col] = None

    # Write the data to CSV
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(rows)

    CONSOLE_LOGGER.info("Data successfully written to %s", csv_file_path)


def process_json(
    output_json_file_path: str, 
    csv_output: bool = False, 
    csv_file_path: bool = None
) -> list[dict[str, float]]:
    """Convert the output JSON file from the ABC model API to a CSV file.

    Args:
        output_json_file_path (str): The path to the output JSON file.
        csv_output (bool): Whether to write the data to a CSV file.
        csv_file_path (str): The path to the CSV file to write the data to.

    Returns:
        list[dict[str, float]]: The data from the JSON file.
    """

    with open(output_json_file_path, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    rows = []

    for result in data["results"]:
        base_data = {"elapsed_time": result["time"], "Met": result["met"], "Clo": result["clo"]}

        overall_data = {
            "Ta": result["overall"]["ta"],
            "MRT": result["overall"]["mrt"],
            "RH": result["overall"]["rh"],
            "Velocity": result["overall"]["v"],
            "Solar": result["overall"]["solar"],
            "Overall_Comfort": result["overall"]["comfort"],
            "Overall_Comfort_weighted": result["overall"]["comfort_weighted"],
            "Overall_Sensation": result["overall"]["sensation"],
            "Overall_Sensation_Linear": result["overall"]["sensation_linear"],
            "Overall_Sensation_Weighted": result["overall"]["sensation_weighted"],
            "MeanSkinTemp": result["overall"]["tskin"],
            "Tblood": result["overall"]["tblood"],
            "Tneutral": result["overall"]["tneutral"],
            "PMV": result["overall"]["pmv"],
            "PPD": result["overall"]["ppd"],
            "EHT": result["overall"]["eht"],
            "Qmet": result["overall"]["q_met"],
            "Qconv": result["overall"]["q_conv"],
            "Qrad": result["overall"]["q_rad"],
            "Qsolar": result["overall"]["q_solar"],
            "Qresp": result["overall"]["q_resp"],
            "Qsweat": result["overall"]["q_sweat"],
        }
        base_data.update(overall_data)

        for segment, segment_data in result["segments"].items():
            segment_data_prefixed = {
                f"Tskin-{segment}": segment_data["tskin"],
                f"Tcore-{segment}": segment_data["tcore"],
                f"Sens-{segment}": segment_data["sensation"],
                f"Sens_weighted-{segment}": segment_data["sensation_weighted"],
                f"Comfort-{segment}": segment_data["comfort"],
                f"Comfort_weighted-{segment}": segment_data["comfort_weighted"],
                f"EHT-{segment}": segment_data["eht"],
                f"Tskin_set-{segment}": segment_data["tskin_set"],
                f"Tskin_set_reg-{segment}": segment_data["tskin_set_reg"],
            }
            base_data.update(segment_data_prefixed)

        rows.append(base_data)

    if csv_output and csv_file_path:
        # Define the desired column order
        column_order = [
            "elapsed_time",
            "Ta",
            "MRT",
            "RH",
            "Velocity",
            "Solar",
            "Clo",
            "Met",
            "Overall_Comfort",
            "Overall_Comfort_weighted",
            "Overall_Sensation",
            "Overall_Sensation_Linear",
            "Overall_Sensation_Weighted",
            "MeanSkinTemp",
            "Tblood",
            "Tneutral",
            "PMV",
            "PPD",
            "EHT",
            "Qmet",
            "Qconv",
            "Qrad",
            "Qsolar",
            "Qresp",
            "Qsweat",
        ]

        # Add segment columns to the order
        segments = [
            "Head",
            "Chest",
            "Back",
            "Pelvis",
            "Left Upper Arm",
            "Right Upper Arm",
            "Left Lower Arm",
            "Right Lower Arm",
            "Left Hand",
            "Right Hand",
            "Left Thigh",
            "Right Thigh",
            "Left Lower Leg",
            "Right Lower Leg",
            "Left Foot",
            "Right Foot",
        ]

        for segment in segments:
            column_order.extend(
                [
                    f"Tskin-{segment}",
                    f"Tcore-{segment}",
                    f"Sens-{segment}",
                    f"Sens_weighted-{segment}",
                    f"Comfort-{segment}",
                    f"Comfort_weighted-{segment}",
                    f"EHT-{segment}",
                    f"Tskin_set-{segment}",
                    f"Tskin_set_reg-{segment}",
                ]
            )

        # Ensure all columns are present in each row
        for row in rows:
            for col in column_order:
                if col not in row:
                    row[col] = None

        # Write the data to CSV
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=column_order)
            writer.writeheader()
            writer.writerows(rows)

        CONSOLE_LOGGER.info("Data successfully written to %s", csv_file_path)

    return rows


def run_abc(
    input_json_file: Path,
    output_json_file: Path,
    output_csv_file: Path,
    api_url="https://fastabc-57h9n.ondigitalocean.app/abc",
):
    """
    Function to run ABC model with your input json file.
    """

    if not input_json_file.exists():
        raise FileNotFoundError(f"Input json file not found: {input_json_file}")

    if not input_json_file.suffix == ".json":
        raise ValueError(f"Input json file must have .json extension: {input_json_file}")

    call_abc_model_api(
        input_file=input_json_file,
        output_file=output_json_file,
        url=api_url,
    )
    output_json_to_csv(output_json_file_path=output_json_file, csv_file_path=output_csv_file)


def make_strings_unique(strings: list[str]) -> list[str]:
    """
    Function to make a list of strings unique by adding a number to the end of each string.
    """
    # create a dictionary from the set of strings
    # the dictionary will store the number of times each string has been seen
    # if a string is already in the dictionary, add a number to the end of the string
    # and increment the number in the dictionary

    d = {}
    for s in set(strings):
        d[s] = 0

    unique_strings = []
    for s in strings:
        d[s] += 1
        if d[s] == 0:
            ns = s
        else:
            ns = f"{s}_{d[s] - 1}"
        unique_strings.append(ns)

    for n, us in enumerate(unique_strings):
        if us.endswith("_0"):
            unique_strings[n] = us[:-2]

    return unique_strings
