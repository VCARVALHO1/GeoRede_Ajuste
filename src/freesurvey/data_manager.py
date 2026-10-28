"""Central data loading, validation and normalization for station folders."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)
CHECK_POINT_CODES = {"ICP", "CHECK", "CHECKPOINT", "CHECK_POINT", "INDEPENDENT_CHECK", "VALIDATION"}
CONTROL_POINT_CODES = {"CONTROL", "CTRL", "CP", "GCP", "CONTROL_POINT"}
SURVEY_POINT_CODES = {"SURVEY", "TARGET", "POINT"}


class SurveyData:
    """Container for normalized station-folder data and legacy measurements."""

    def __init__(self):
        self.measurements: List[Dict] = []
        self.all_points: List[str] = []
        self.target_points: List[str] = []

    def add_measurement(self, measurement: Dict):
        self.measurements.append(measurement)


def extract_station_number(name) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 999999


def split_station_line(line: str, delimiter: str = "tab") -> List[str]:
    delimiter = (delimiter or "tab").lower()
    if delimiter == "auto":
        for candidate in ("\t", ";", ",", None):
            parts = line.split(candidate) if candidate else line.split()
            if len(parts) >= 4:
                return [part.strip() for part in parts]
        return [line.strip()]
    if delimiter == "semicolon":
        return [part.strip() for part in line.split(";")]
    if delimiter == "comma":
        return [part.strip() for part in line.split(",")]
    if delimiter == "space":
        return [part.strip() for part in line.split()]
    return [part.strip() for part in line.split("\t")]


def parse_ordered_coordinates(
    coordinate_values: Iterable[str],
    coordinate_order: str = "X,Y,Z",
) -> Tuple[float, float, float]:
    labels = [label.strip().upper() for label in coordinate_order.split(",")]
    if len(labels) != 3 or set(labels) != {"X", "Y", "Z"}:
        labels = ["X", "Y", "Z"]
    values_by_label = {
        label: float(str(value).strip().replace(",", "."))
        for label, value in zip(labels, coordinate_values)
    }
    return values_by_label["X"], values_by_label["Y"], values_by_label["Z"]


def normalize_point_code(code: str | None) -> str:
    return str(code or "").strip().upper().replace("-", "_").replace(" ", "_")


def point_role(point_id: str, point_code: str | None = None) -> str:
    code = normalize_point_code(point_code)
    if code in CHECK_POINT_CODES:
        return "icp"
    if code in CONTROL_POINT_CODES:
        return "control"
    if code in SURVEY_POINT_CODES:
        return "survey"
    text = str(point_id).strip().upper()
    if text.startswith("ICP"):
        return "icp"
    if text.startswith("CP"):
        return "control"
    return "survey"


def normalize_station(
    name: str,
    points: Dict[str, Tuple[float, float, float]],
    coordinate_order: str = "X,Y,Z",
    point_codes: Dict[str, str] | None = None,
    point_comments: Dict[str, str] | None = None,
) -> Dict:
    point_codes = point_codes or {}
    point_comments = point_comments or {}
    all_ids = list(points.keys())
    coords = np.asarray([points[pid] for pid in all_ids], dtype=float)
    roles = {pid: point_role(pid, point_codes.get(pid)) for pid in all_ids}
    ctrl_ids = [pid for pid in all_ids if roles[pid] == "control"]
    icp_ids = [pid for pid in all_ids if roles[pid] == "icp"]
    station = {
        "name": name,
        "number": extract_station_number(name),
        "coords": coords,
        "raw_coords": coords.copy(),
        "original_coords": coords.copy(),
        "coords_original": coords.copy(),
        "all_ids": all_ids,
        "ctrl_ids": ctrl_ids,
        "icp_ids": icp_ids,
        "point_codes": {
            pid: (point_codes.get(pid) or ("ICP" if pid in icp_ids else ("CONTROL" if pid in ctrl_ids else "SURVEY")))
            for pid in all_ids
        },
        "point_comments": {
            pid: comment for pid, comment in point_comments.items() if comment
        },
        "coordinate_order": coordinate_order,
    }
    return station


def apply_independent_check_points(stations: List[Dict], point_ids: Iterable[str] | None) -> List[Dict]:
    selected = {str(point_id) for point_id in (point_ids or []) if str(point_id).strip()}
    if not selected:
        return stations
    for station in stations:
        all_ids = {str(point_id) for point_id in station.get("all_ids", [])}
        station_icp = {str(point_id) for point_id in station.get("icp_ids", [])}
        station_icp |= selected & all_ids
        station["icp_ids"] = sorted(
            station_icp,
            key=lambda point_id: (extract_station_number(point_id), point_id),
        )
        station["ctrl_ids"] = [
            point_id for point_id in station.get("ctrl_ids", [])
            if str(point_id) not in station_icp
        ]
        codes = dict(station.get("point_codes", {}))
        for point_id in station["icp_ids"]:
            codes[point_id] = "ICP"
        station["point_codes"] = codes
    return stations


class DataManager:
    """Single entry point for input data used by GUI, CLI and analyzers."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def read_station_file(
        self,
        file_path: str | Path,
        delimiter: str = "tab",
        coordinate_order: str = "X,Y,Z",
    ) -> Dict[str, Tuple[float, float, float]]:
        points, _, _ = self.read_station_file_records(
            file_path,
            delimiter=delimiter,
            coordinate_order=coordinate_order,
        )
        return points

    def read_station_file_records(
        self,
        file_path: str | Path,
        delimiter: str = "tab",
        coordinate_order: str = "X,Y,Z",
    ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, str], Dict[str, str]]:
        file_path = Path(file_path)
        points: Dict[str, Tuple[float, float, float]] = {}
        point_codes: Dict[str, str] = {}
        point_comments: Dict[str, str] = {}
        with file_path.open("r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                parts = split_station_line(line, delimiter)
                if line_number == 1 and parts and parts[0].lower() in {"ptid", "point", "id"}:
                    continue
                if len(parts) < 4:
                    self.logger.warning("Skipping malformed line %s in %s", line_number, file_path.name)
                    continue
                point_id = parts[0].strip()
                if not point_id:
                    continue
                try:
                    points[point_id] = parse_ordered_coordinates(parts[1:4], coordinate_order)
                    if len(parts) >= 5 and parts[4]:
                        point_codes[point_id] = parts[4].strip().upper()
                    if len(parts) >= 6:
                        comment = " ".join(part for part in parts[5:] if part).strip()
                        if comment:
                            point_comments[point_id] = comment
                except ValueError:
                    self.logger.debug("Skipping non-numeric point %s in %s", point_id, file_path.name)
        return points, point_codes, point_comments

    def load_station_folder(
        self,
        folder_path: str | Path,
        delimiter: str = "tab",
        coordinate_order: str = "X,Y,Z",
        independent_check_points: Iterable[str] | None = None,
    ) -> SurveyData:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Station data folder not found: {folder}")

        files = sorted(
            list(folder.glob("*.TXT")) + list(folder.glob("*.txt")),
            key=lambda path: extract_station_number(path.stem),
        )
        if not files:
            raise ValueError(f"No .TXT files found in folder: {folder}")

        stations = []
        for txt_file in files:
            points, point_codes, point_comments = self.read_station_file_records(
                txt_file,
                delimiter=delimiter,
                coordinate_order=coordinate_order,
            )
            if not points:
                self.logger.warning("No valid points found in %s", txt_file.name)
                continue
            stations.append(normalize_station(
                txt_file.stem,
                points,
                coordinate_order,
                point_codes=point_codes,
                point_comments=point_comments,
            ))

        apply_independent_check_points(stations, independent_check_points)
        return self.to_survey_data(stations)

    def to_survey_data(self, stations: List[Dict]) -> SurveyData:
        survey_data = SurveyData()
        station_data_map = {}
        all_control_points = {}

        for station in sorted(stations, key=lambda item: item.get("number", 999999)):
            all_ids = list(station.get("all_ids", []))
            coords = np.asarray(station.get("coords", []), dtype=float)
            ctrl_ids = set(station.get("ctrl_ids", []))
            icp_ids = set(station.get("icp_ids", []))
            control_points = {}
            survey_points = {}
            for point_id, coord in zip(all_ids, coords):
                if point_id in ctrl_ids:
                    control_points[point_id] = tuple(coord)
                    all_control_points[point_id] = tuple(coord)
                else:
                    survey_points[point_id] = tuple(coord)
            station_data_map[station["name"]] = {
                "control_points": control_points,
                "survey_points": survey_points,
                "icp_points": {
                    point_id: survey_points[point_id]
                    for point_id in icp_ids
                    if point_id in survey_points
                },
            }

        for station_name, data in station_data_map.items():
            for cp_id, cp_coords in data["control_points"].items():
                for sp_id, sp_coords in data["survey_points"].items():
                    dx = sp_coords[0] - cp_coords[0]
                    dy = sp_coords[1] - cp_coords[1]
                    dz = sp_coords[2] - cp_coords[2]
                    survey_data.add_measurement({
                        "from_point": cp_id,
                        "to_point": sp_id,
                        "angle_hz": np.arctan2(dy, dx),
                        "angle_v": np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2)),
                        "distance": float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)),
                        "height_instrument": 0.0,
                        "height_target": 0.0,
                        "station": station_name,
                    })
                    for point in (cp_id, sp_id):
                        if point not in survey_data.all_points:
                            survey_data.all_points.append(point)

        sorted_stations = sorted(stations, key=lambda item: item.get("number", 999999))
        survey_data.stations = {station["name"]: station for station in sorted_stations}
        survey_data.control_points = all_control_points
        survey_data.target_points = sorted({
            point_id
            for data in station_data_map.values()
            for point_id in data["survey_points"].keys()
        })
        return survey_data

    def validate_survey_data(self, survey_data: SurveyData) -> bool:
        stations = getattr(survey_data, "stations", {})
        if not stations:
            self.logger.error("No stations loaded")
            return False
        valid = True
        for station_name, station in stations.items():
            coords = np.asarray(station.get("coords", []), dtype=float)
            all_ids = list(station.get("all_ids", []))
            if coords.ndim != 2 or coords.shape[1] != 3:
                self.logger.error("Station %s does not have Nx3 coordinates", station_name)
                valid = False
            if len(all_ids) != len(coords):
                self.logger.error("Station %s has mismatched all_ids/coords length", station_name)
                valid = False
            if len(station.get("ctrl_ids", [])) < 1:
                self.logger.warning("Station %s has no processing control points", station_name)
        return valid


__all__ = [
    "DataManager",
    "SurveyData",
    "apply_independent_check_points",
    "extract_station_number",
    "normalize_station",
    "normalize_point_code",
    "parse_ordered_coordinates",
    "point_role",
    "split_station_line",
]
