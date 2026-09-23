"""Hand-crafted schedules designed to trigger one constraint at a time."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .checks import schedule_fingerprint
from .model import Scenario


def valid_q3_schedule(scenario: Scenario) -> dict[str, Any]:
    return {
        "sorties": [
            {
                "sortie_id": "P1",
                "uav_id": "U01",
                "uav_type": "A",
                "battery_id": "A-BAT-01",
                "route": ["O01", "S001", "O01"],
                "box_ids": ["S001-MED-01", "S001-WAT-01"],
                "node_arrival_times": [0.0, 150.0, 350.0],
                "relay_id": None,
                "start_time": 0.0,
                "end_time": 350.0,
                "node_events": [
                    {"node": "O01", "arrival": 0.0, "departure": 30.0},
                    {"node": "S001", "arrival": 150.0, "departure": 250.0},
                    {"node": "O01", "arrival": 350.0, "departure": 350.0},
                ],
                "leg_loads": [17.0, 0.0],
                "leg_energy": [0.5, 0.4],
                "trip_energy": 0.9,
                "return_soc": 0.8,
                "delivery_times": {"S001-MED-01": 250.0, "S001-WAT-01": 250.0},
                "communication_intervals": [{"start": 0.0, "end": 350.0, "state": "direct"}],
            },
            {
                "sortie_id": "P2",
                "uav_id": "U05",
                "uav_type": "B",
                "battery_id": "B-BAT-01",
                "route": ["O01", "S002", "O01"],
                "box_ids": ["S002-MED-01", "S002-WAT-01"],
                "node_arrival_times": [0.0, 150.0, 350.0],
                "relay_id": None,
                "start_time": 0.0,
                "end_time": 350.0,
                "node_events": [
                    {"node": "O01", "arrival": 0.0, "departure": 30.0},
                    {"node": "S002", "arrival": 150.0, "departure": 250.0},
                    {"node": "O01", "arrival": 350.0, "departure": 350.0},
                ],
                "leg_loads": [17.0, 0.0],
                "leg_energy": [0.4, 0.3],
                "trip_energy": 0.7,
                "return_soc": 0.825,
                "delivery_times": {"S002-MED-01": 250.0, "S002-WAT-01": 250.0},
                "communication_intervals": [{"start": 0.0, "end": 350.0, "state": "direct"}],
            },
            {
                "sortie_id": "P3",
                "uav_id": "U07",
                "uav_type": "C",
                "battery_id": "C-BAT-01",
                "route": ["O01", "S001", "S015", "O01"],
                "node_arrival_times": [500.0, 650.0, 800.0, 1000.0],
                "relay_id": "R01",
                "start_time": 500.0,
                "end_time": 1000.0,
                "node_events": [
                    {"node": "O01", "arrival": 500.0, "departure": 530.0},
                    {"node": "S001", "arrival": 650.0, "departure": 700.0},
                    {"node": "S015", "arrival": 800.0, "departure": 850.0},
                    {"node": "O01", "arrival": 1000.0, "departure": 1000.0},
                ],
                "box_ids": ["S001-FOD-01", "S001-HYG-01", "S001-HYG-02", "S015-MED-01"],
                "leg_loads": [23.0, 3.0, 0.0],
                "leg_energy": [0.6, 0.5, 0.4],
                "trip_energy": 1.5,
                "return_soc": 0.8125,
                "delivery_times": {
                    "S001-FOD-01": 700.0,
                    "S001-HYG-01": 700.0,
                    "S001-HYG-02": 700.0,
                    "S015-MED-01": 850.0,
                },
                "communication_intervals": [
                    {"start": 500.0, "end": 1000.0, "state": "relay", "relay_id": "R01"}
                ],
            },
        ],
        "relay_tasks": [
            {
                "task_id": "RT1",
                "relay_id": "R01",
                "component_id": "R-COMP-01",
                "start": 450.0,
                "end": 1100.0,
                "service_start": 500.0,
                "service_end": 1000.0,
                "energy": 0.6,
                "return_soc": 0.8125,
                "services": ["S001", "S015"],
                "position": [500.0, 500.0],
                "height_m": 200.0,
            }
        ],
    }


def valid_q4_partition(schedule: dict[str, Any]) -> dict[str, Any]:
    partition = {
        "group_count": 2,
        "service_to_group": {"S001": 1, "S002": 2, "S015": 1},
        "group_resource_counts": {
            "1": {"A_uav": 1, "C_uav": 1, "A_battery": 1, "C_battery": 1, "R_relay": 1, "R_component": 1},
            "2": {"B_uav": 1, "B_battery": 1},
        },
        "group_resource_assignments": {
            "1": {"uav": ["U01", "U07"], "battery": ["A-BAT-01", "C-BAT-01"], "relay": ["R01"], "component": ["R-COMP-01"]},
            "2": {"uav": ["U05"], "battery": ["B-BAT-01"], "relay": [], "component": []},
        },
        "resource_shortages": {
            "1": {"A_uav": 0, "C_uav": 0, "A_battery": 0, "C_battery": 0, "R_relay": 0, "R_component": 0},
            "2": {"B_uav": 0, "B_battery": 0},
        },
    }
    partition["q3_schedule_fingerprint"] = schedule_fingerprint(schedule)
    return partition


def invalid_cases(scenario: Scenario, schedule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    volume = deepcopy(schedule)
    volume["sorties"][0]["box_ids"] = ["S001-FOD-01", "S001-HYG-01"]
    volume["sorties"][2]["box_ids"] = ["S001-MED-01", "S001-WAT-01", "S001-HYG-02", "S015-MED-01"]
    volume["sorties"][2]["leg_loads"] = [37.0, 3.0, 0.0]
    cases["mutant_volume_constraint"] = {"kind": "q3", "schedule": volume, "expected_codes": ["VOLUME_EXCEEDED"]}

    battery = deepcopy(schedule)
    battery["sorties"][1]["uav_id"] = "U01"
    battery["sorties"][1]["uav_type"] = "A"
    battery["sorties"][1]["battery_id"] = "A-BAT-01"
    battery["sorties"][1]["start_time"] = 100.0
    battery["sorties"][1]["end_time"] = 450.0
    cases["mutant_battery_overlap"] = {"kind": "q3", "schedule": battery, "expected_codes": ["UAV_OVERLAP", "BATTERY_OVERLAP"]}

    comm_gap = deepcopy(schedule)
    comm_gap["sorties"][2]["communication_intervals"] = [
        {"start": 500.0, "end": 700.0, "state": "relay", "relay_id": "R01"},
        {"start": 750.0, "end": 1000.0, "state": "relay", "relay_id": "R01"},
    ]
    cases["mutant_communication_gap"] = {"kind": "q3", "schedule": comm_gap, "expected_codes": ["COMMUNICATION_GAP"]}

    load_trace = deepcopy(schedule)
    load_trace["sorties"][2]["leg_loads"] = [23.0, 23.0, 0.0]
    cases["mutant_load_trace"] = {"kind": "q3", "schedule": load_trace, "expected_codes": ["LOAD_TRACE"]}

    cross_resource = deepcopy(schedule)
    cross_resource["sorties"][1]["uav_id"] = "U01"
    cross_resource["sorties"][1]["uav_type"] = "A"
    cross_resource["sorties"][1]["battery_id"] = "A-BAT-01"
    partition = valid_q4_partition(cross_resource)
    partition["service_to_group"] = {"S001": 1, "S002": 2, "S015": 1}
    partition["group_resource_assignments"]["2"]["uav"] = ["U01"]
    partition["group_resource_assignments"]["2"]["battery"] = ["A-BAT-01"]
    cases["mutant_cross_group_resource"] = {"kind": "q4", "schedule": cross_resource, "partition": partition, "expected_codes": ["CROSS_GROUP_RESOURCE"]}

    split = valid_q4_partition(schedule)
    split["service_to_group"] = {"S001": 1, "S002": 2, "S015": 2}
    cases["mutant_split_route_group"] = {"kind": "q4", "schedule": schedule, "partition": split, "expected_codes": ["ROUTE_SPLIT"]}
    return cases
