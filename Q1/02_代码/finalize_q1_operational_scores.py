#!/usr/bin/env python3
"""Retired Q1 exporter.

The old q_equal closeout exporter has been superseded by the frozen
Q1-q_huber-v1 interface. Use results/q1_final/export_q1_final_interface.py.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This exporter is retired. Use "
        "results/q1_final/export_q1_final_interface.py for Q1-q_huber-v1."
    )


if __name__ == "__main__":
    main()
