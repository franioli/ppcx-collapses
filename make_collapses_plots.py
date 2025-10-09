from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shapely import contains_xy
from shapely import wkt as shapely_wkt
from sqlalchemy import create_engine
from tqdm import tqdm

from ppcollapse import setup_logger
from ppcollapse.utils.config import ConfigManager
from ppcollapse.utils.database import (
    fetch_dic_analysis_ids,
    get_collapses_df,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
)

# Use Agg backend for script (non-interactive)
matplotlib.use("Agg")

logger = setup_logger(level="WARNING", name="ppcx")

# -------------------------
# PARAMETERS (edit here)
# -------------------------
CONFIG_PATH: str | Path = "config.yaml"
DAYS_BEFORE = 10
OUTPUT_DIR = Path("output/collapses_timeseries")
N_JOBS = 6  # number of parallel jobs
VELOCITY_YLIM = (0, 20)  # fixed y axis limits for velocity plot, or None
# -------------------------


def fetch_dic_before(
    config: ConfigManager, collapse_date: str, days_before: int, **kwargs
):
    """Fetch DIC analyses in the window [collapse_date - days_before, collapse_date]."""

    engine = create_engine(config.db_url)
    start_date = pd.to_datetime(collapse_date) - pd.Timedelta(days=days_before)
    start_date_str = start_date.strftime("%Y-%m-%d")
    dic_ids = fetch_dic_analysis_ids(
        db_engine=engine,
        reference_date_start=start_date_str,
        reference_date_end=collapse_date,
        **kwargs,
    )
    if len(dic_ids) == 0:
        return pd.DataFrame(), {}
    dic_metadata = get_dic_analysis_by_ids(dic_ids=dic_ids, db_engine=engine)
    dic_data = get_multi_dic_data(dic_ids=dic_ids, config=config, stack_results=False)
    return dic_metadata, dic_data


def compute_dic_stats_for_geom(
    geom, dic_metadata: pd.DataFrame, dic_data: dict
) -> pd.DataFrame:
    """Return a dataframe indexed by dic_id with stats (date, n_points, mean,std,min,max,median)."""
    rows = []
    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue
        mask = contains_xy(geom, pts["x"].to_numpy(), pts["y"].to_numpy())
        sel = pts.loc[mask]
        vals = pd.to_numeric(sel["V"], errors="coerce")

        cur_date = dic_metadata.loc[dic_metadata.dic_id == dic_id, "reference_date"]
        if not cur_date.empty:
            date = pd.to_datetime(cur_date.values[0])
        else:
            date = None
        rows.append(
            {
                "dic_id": dic_id,
                "date": date,
                "n_points": int(len(sel)),
                "mean": float(vals.mean()) if not vals.empty else np.nan,
                "std": float(vals.std()) if not vals.empty else np.nan,
                "min": float(vals.min()) if not vals.empty else np.nan,
                "max": float(vals.max()) if not vals.empty else np.nan,
                "median": float(vals.median()) if not vals.empty else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dic_id",
                "date",
                "n_points",
                "mean",
                "std",
                "min",
                "max",
                "median",
            ]
        ).set_index("dic_id")
    df = pd.DataFrame(rows).set_index("dic_id")
    df = df.sort_values("date")
    return df


def make_collapse_plot(
    stats_df: pd.DataFrame,
    collapse_row: pd.Series,
    image: np.ndarray,
    *,
    velocity_ylim: tuple[int, int] | None = VELOCITY_YLIM,
) -> tuple[Figure, Any]:
    """Create the two-panel (image + timeseries) plot for a collapse and save it.

    Returns the figure and axes objects.
    """
    collapse_id = int(collapse_row["id"])
    geom = shapely_wkt.loads(collapse_row["geom_wkt"])
    xs, ys = shapely_wkt.loads(shapely_wkt.dumps(geom)).exterior.xy
    date_ts = pd.to_datetime(collapse_row["date"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    ax_img, ax_ts = axes

    # Left: image + geometry
    ax_img.imshow(image)
    ax_img.plot(xs, ys, color="red", linewidth=2)
    ax_img.fill(xs, ys, facecolor="none", edgecolor="red", alpha=0.6)
    ax_img.set_axis_off()

    # Right: timeseries
    if stats_df is None or stats_df.empty:
        ax_ts.text(0.5, 0.5, "No DIC data inside geometry", ha="center", va="center")
    else:
        stats = stats_df.copy()
        stats["date"] = pd.to_datetime(stats["date"])
        for col in ["n_points", "mean", "std", "min", "max", "median"]:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors="coerce")
        x = stats["date"]
        if "std" in stats.columns and "mean" in stats.columns:
            y1 = stats["mean"] - stats["std"]
            y2 = stats["mean"] + stats["std"]
            ax_ts.fill_between(x, y1, y2, color="gray", alpha=0.25, label="±1 std")
        if "mean" in stats.columns:
            ax_ts.plot(x, stats["mean"], marker="o", label="Mean")
        if "median" in stats.columns:
            ax_ts.plot(x, stats["median"], marker="", label="Median", linewidth=0.5)

        if velocity_ylim is not None and len(velocity_ylim) == 2:
            ax_ts.set_ylim(velocity_ylim)
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Velocity [px/day]")
        ax_ts.legend()
        ax_ts.grid(alpha=0.3)
        fig.autofmt_xdate()

    area = collapse_row.get("area", float("nan"))
    volume = collapse_row.get("volume", float("nan"))
    fig.suptitle(
        f"{date_ts.strftime('%Y-%m-%d')}\nArea {area:.1f} m², Volume {volume:.1f} m³\nCollapse ID {collapse_id}"
    )
    fig.tight_layout()

    return fig, axes


def process_collapse(
    collapse_row: pd.Series,
    cfg: ConfigManager,
    days_before: int,
    out_dir: Path,
) -> Optional[Path]:
    """
    Make the two-panel plot (image + geometry on left, velocity timeseries on right)
    for one collapse, reusing compute_dic_stats_for_geom and fetch_dic_before.
    """
    collapse_id = int(collapse_row["id"])
    date_ts = pd.to_datetime(collapse_row["date"])
    collapse_date = date_ts.date()
    geom_wkt = collapse_row["geom_wkt"]
    logger.info(f"Processing collapse id={collapse_id} date={collapse_date}")

    try:
        geom = shapely_wkt.loads(geom_wkt)
    except Exception as exc:
        logger.error(f"Invalid WKT for collapse {collapse_id}: {exc}")
        return None

    # fetch image for left panel (may fail separately)
    image = None
    try:
        image = get_image(image_id=int(collapse_row["image_id"]), config=cfg)
    except Exception as exc:
        logger.error(f"Failed to fetch image for collapse {collapse_id}: {exc}")
        return None

    # fetch DIC data and compute stats inside geometry
    try:
        dic_meta, dic_data = fetch_dic_before(
            config=cfg,
            collapse_date=collapse_date.isoformat(),
            days_before=days_before,
            camera_name="PPCX_Tele",
            dt_hours_min=72,
            dt_hours_max=96,
        )
        if dic_meta.empty or not dic_data:
            logger.warning(f"No DIC data found before collapse {collapse_id}")
            return None
    except Exception as exc:
        logger.exception(
            f"Error fetching or computing DIC stats for collapse {collapse_id}: {exc}"
        )
        return None

    # Compute statistics for DIC points inside geometry
    stats_df = compute_dic_stats_for_geom(geom, dic_meta, dic_data)

    # Build figure
    try:
        fig, ax = make_collapse_plot(
            stats_df=stats_df,
            collapse_row=collapse_row,
            image=np.asarray(image),
            velocity_ylim=VELOCITY_YLIM,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{collapse_date.isoformat()}_collapse_{collapse_id}_timeseries{DAYS_BEFORE}days.jpg"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved plot for collapse {collapse_id} -> {out_path}")

        return out_path

    except Exception:
        logger.exception(
            "Failed to build/save plot for collapse row: %s", collapse_row.to_dict()
        )
        return None

    out_path = out_dir / f"collapse_{collapse_id}_timeseries_placeholder.txt"
    return out_path


def main() -> bool:
    cfg = ConfigManager(CONFIG_PATH)

    # NOTE: TEMPORARY OVERRIDE DB NAME TO SANDBOX
    cfg.set("database.name", "sandbox")
    cfg.set("database.password", "postgresppcx")

    engine = create_engine(cfg.db_url)
    df = get_collapses_df(engine)
    if df.empty:
        logger.warning("No collapses found in database.")
        return False

    def _process_row(row):
        try:
            process_collapse(
                row,
                cfg=cfg,
                days_before=DAYS_BEFORE,
                out_dir=OUTPUT_DIR,
            )
        except Exception:
            logger.exception(
                f"Unexpected error plotting collapse id={int(row.get('id', -1))}"
            )

    with Parallel(n_jobs=N_JOBS, backend="threading") as parallel:
        parallel(
            delayed(_process_row)(row)
            for _, row in tqdm(
                df.iterrows(), total=df.shape[0], desc="Processing collapses"
            )
        )

    return True


if __name__ == "__main__":
    main()
