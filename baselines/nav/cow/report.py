"""Tabulate one CoW experiment (the shared report with ``--method cow``)."""

from __future__ import annotations

from ..experiment.report import main as report_main


def main(argv: list[str] | None = None) -> int:
    return report_main(argv, method="cow")


if __name__ == "__main__":
    raise SystemExit(main())
