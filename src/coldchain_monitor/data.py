"""Temperature logs: container, CSV input/output and a synthetic generator.

All series in this package are sampled at a fixed 5-minute step, the
resolution used in the thesis experiments.

The synthetic generator exists so that the tests, the demo and the CI
pipeline can run without downloading anything. It imitates a refrigerated
milk store (thermostat cycling, door openings, occasional cooling faults,
a coarse 0.5 degC sensor), but it is **not** a substitute for real sensor
data: results obtained on it show that the software works, not how the
methods rank on a real cold chain.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: sampling step of every series, minutes
STEP_MINUTES = 5
#: number of samples per hour
STEPS_PER_HOUR = 60 // STEP_MINUTES

CSV_COLUMNS = ("time_min", "temperature_c")


@dataclass(frozen=True)
class TemperatureLog:
    """One temperature series sampled every :data:`STEP_MINUTES` minutes."""

    temperature: np.ndarray  #: degC
    name: str = "log"

    def __post_init__(self) -> None:
        values = np.asarray(self.temperature, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("a log needs at least two readings in a 1-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("temperature readings must be finite numbers")
        object.__setattr__(self, "temperature", values)

    def __len__(self) -> int:
        return int(self.temperature.size)

    @property
    def time_min(self) -> np.ndarray:
        """Time of each reading in minutes from the start of the log."""
        return np.arange(len(self)) * float(STEP_MINUTES)

    @property
    def hours(self) -> float:
        """Duration of the log in hours."""
        return (len(self) - 1) * STEP_MINUTES / 60.0


def synthetic_log(
    hours: float = 168.0,
    seed: int | None = 0,
    setpoint: float = 2.5,
    band: float = 0.6,
    quantization: float = 0.5,
    noise: float = 0.05,
    doors_per_day: float = 2.0,
    faults_per_week: float = 4.0,
    name: str | None = None,
) -> TemperatureLog:
    """Simulate a refrigerated store controlled by an on/off thermostat.

    The compressor switches on above ``setpoint + band`` and off below
    ``setpoint - band``. Door openings add heat for 10-30 minutes; a cooling
    fault lasts 3-8 hours, during which the store warms steadily and usually
    drifts above the 4 degC limit. Readings are rounded to
    ``quantization`` degC (0 disables rounding).
    """
    if hours <= 0:
        raise ValueError("hours must be positive")
    if band <= 0 or quantization < 0 or noise < 0:
        raise ValueError("band must be positive; quantization and noise non-negative")
    rng = np.random.default_rng(seed)
    n = int(round(hours * STEPS_PER_HOUR)) + 1
    dt = STEP_MINUTES / 60.0  # hours
    warm_rate, cool_rate, door_rate, fault_rate = 0.15, 0.5, 2.5, 0.4  # degC per hour

    door_left = np.zeros(n, dtype=int)
    for start in np.nonzero(rng.random(n) < doors_per_day / (24 * STEPS_PER_HOUR))[0]:
        door_left[start : start + int(rng.integers(2, 7))] = 1
    fault = np.zeros(n, dtype=bool)
    for start in np.nonzero(rng.random(n) < faults_per_week / (7 * 24 * STEPS_PER_HOUR))[0]:
        fault[start : start + int(rng.integers(3 * STEPS_PER_HOUR, 8 * STEPS_PER_HOUR))] = True

    true = np.empty(n)
    temperature = setpoint + rng.uniform(-band, band)
    compressor_on = False
    for i in range(n):
        true[i] = temperature
        if temperature > setpoint + band:
            compressor_on = True
        elif temperature < setpoint - band:
            compressor_on = False
        if fault[i]:
            rate = fault_rate
        elif compressor_on:
            rate = -cool_rate
        else:
            rate = warm_rate
        if door_left[i]:
            rate += door_rate
        temperature += rate * dt

    readings = true + rng.normal(0.0, noise, n)
    if quantization > 0:
        readings = np.round(readings / quantization) * quantization
    label = name if name is not None else f"synthetic-{seed}"
    return TemperatureLog(readings, label)


def write_csv(log: TemperatureLog, path: str | Path) -> Path:
    """Write a log to CSV with the columns :data:`CSV_COLUMNS`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for minute, value in zip(log.time_min, log.temperature, strict=True):
            writer.writerow([f"{minute:.0f}", f"{value:.2f}"])
    return path


def read_csv(path: str | Path) -> TemperatureLog:
    """Read a log from CSV.

    Only the ``temperature_c`` column is required. Rows must already be on a
    regular 5-minute grid, in time order; resample raw exports first.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "temperature_c" not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: missing column temperature_c")
        rows = list(reader)
    try:
        values = np.array([float(row["temperature_c"]) for row in rows])
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric temperature value ({exc})") from exc
    if values.size < 2:
        raise ValueError(f"{path}: file contains fewer than two readings")
    return TemperatureLog(values, path.stem)
