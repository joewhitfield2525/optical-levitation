from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("/Users/josephwhitfield/Masters/Summer Project/Power_02min.csv")


def load_power_data(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    try:
        graph_data_index = next(
            index for index, row in enumerate(rows) if row and row[0] == "--Graph Data--"
        )
    except StopIteration as exc:
        raise ValueError("Could not find '--Graph Data--' section in the CSV file.") from exc

    header_index = graph_data_index + 1
    data_rows = rows[header_index + 1 :]

    time_ms = []
    power_w = []

    for row in data_rows:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue

        try:
            time_ms.append(float(row[0]))
            power_w.append(float(row[1]))
        except ValueError:
            continue

    if not time_ms:
        raise ValueError("No numeric power data was found in the CSV file.")

    return np.array(time_ms), np.array(power_w)


def main():
    time_ms, power_w = load_power_data(CSV_PATH)
    time_s = time_ms / 1000

    print(f"Loaded: {CSV_PATH}")
    print(f"Samples: {len(power_w)}")
    print(f"Mean power: {np.mean(power_w):.6g} W")
    print(f"Standard deviation: {np.std(power_w, ddof=1):.6g} W")

    plt.figure(figsize=(12, 6))
    plt.plot(time_s, power_w, linewidth=0.9)
    plt.axhline(np.mean(power_w), color="tab:red", linestyle="--", linewidth=1.0, label="Mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power_02min.csv")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
