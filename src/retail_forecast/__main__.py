"""Command line entry point: `python -m retail_forecast`."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .config import ARTIFACTS_DIR, RAW_DIR, WEB_DATA_DIR
from .data import load_panel
from .export import assistant_context, dashboard_payload, write_json
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate and export the data behind the Retail Store dashboard.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=WEB_DATA_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="skip the hyperparameter search (single candidate)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("retail_forecast")
    start = time.perf_counter()

    result = run(load_panel(args.raw_dir), quick=args.quick)
    payload = dashboard_payload(result)
    write_json(payload, args.out_dir / "dashboard_data.json")
    write_json(assistant_context(payload), args.out_dir / "assistant_context.json", indent=2)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    result.tuning_table.to_csv(ARTIFACTS_DIR / "tuning.csv", index=False)
    result.test.to_csv(ARTIFACTS_DIR / "holdout_predictions.csv", index=False)
    write_json(payload["evaluation"] | {"forecast": payload["forecast"]},
               ARTIFACTS_DIR / "metrics.json", indent=2)

    log.info("Model comparison on the holdout:\n%s",
             result.model_table.round(4).to_string(index=False))
    log.info("Done in %.0fs. Wrote %s", time.perf_counter() - start, args.out_dir)


if __name__ == "__main__":
    main()
