from pathlib import Path

import fiona
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from sqlalchemy import create_engine, text
from tqdm import tqdm

from ppcollapse import logger
from ppcollapse.utils.config import ConfigManager
from ppcollapse.utils.database import (
    fetch_image_ids,
    fetch_image_metadata_by_ids,
)

# Load configuration
config = ConfigManager(config_path="config.yaml")
db_engine = create_engine(
    "postgresql://postgres:postgresppcx@150.145.51.193:5434/sandbox"
)

shape_dir = Path("data/SHAPEFILES_adj")
file_ext = ".shp"
volume_file_dir = Path("data/crolli")


def read_shapely_geom_from_file(path: Path, invert_y: bool = False) -> BaseGeometry:
    """Read first feature from a shapefile, flip Y (QGIS -> image coords), return (geom, area, perimeter)."""
    with fiona.open(path) as src:
        polygons = [feature["geometry"] for feature in src]
    if not polygons:
        raise ValueError(f"No features found in {path}")
    poly = polygons[0]
    geom = shape(poly)

    if invert_y:
        # Invert Y axis because the polygons were created in qgis
        geom = transform(
            lambda x, y, z=None: (x, -y) if z is None else (x, -y, z), geom
        )

    return geom


def write_geom_wkt_to_db(
    db_engine,
    image_id: int,
    geom: BaseGeometry,
    area: float | None = None,
    volume: float | None = None,
    table: str = "ppcx_app_collapse",
) -> int:
    """Ensure minimal table exists and insert geometry as WKT (SRID=0). Returns inserted id."""
    wkt = shapely_wkt.dumps(geom)

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id SERIAL PRIMARY KEY,
        image_id INTEGER NOT NULL REFERENCES ppcx_app_image(id),
        geom geometry(Geometry,0),
        area DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """

    insert_sql = f"""
    INSERT INTO {table} (image_id, geom, area, volume)
    VALUES (:image_id, ST_GeomFromText(:wkt, 0), :area, :volume)
    RETURNING id;
    """
    with db_engine.begin() as conn:
        conn.execute(text(create_table_sql))
        res = conn.execute(
            text(insert_sql),
            {
                "image_id": int(image_id),
                "wkt": wkt,
                "area": float(area) if area is not None else None,
                "volume": float(volume) if volume is not None else None,
            },
        )
        return res.scalar_one()


def read_collapse_volume_file(path: Path) -> pd.DataFrame:
    import re

    # regex: 6 mandatory columns then optional trailing flag that starts with %
    # allow numeric type or literal "NaN" for missing type
    pattern = re.compile(
        r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+((?:\d+|NaN))\s+(\d+)(?:\s+(%.*))?\s*$",
        re.IGNORECASE,
    )
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("%"):
            continue
        m = pattern.match(ln)
        if m:
            year, month, day_start, day_end, typ, size, flag = m.groups()
        else:
            parts = ln.split()
            if len(parts) < 6:
                raise ValueError(f"Unparseable line: {ln!r}")
            year, month, day_start, day_end, typ, size = parts[:6]
            flag = None
            if len(parts) > 6:
                tail = " ".join(parts[6:])
                flag = tail if tail.startswith("%") else None

        # normalize type: treat NaN (any case) as missing -> None
        typ_val = None if isinstance(typ, str) and typ.lower() == "nan" else int(typ)

        rows.append(
            {
                "date": pd.to_datetime(
                    f"{year}-{int(month):02d}-{int(day_end):02d}", format="%Y-%m-%d"
                ),
                "year": int(year),
                "month": int(month),
                "day_start": int(day_start),
                "day_end": int(day_end),
                "type": typ_val,
                "volume": int(size),
                "flag_raw": flag,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for year_dir in sorted(shape_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        files = sorted(year_dir.glob(f"*{file_ext}"))
        logger.info(f"Processing year directory: {year_dir}, found {len(files)} files")

        # Check if a volume file exists
        year = year_dir.name.split("_")[0]
        volume_file = volume_file_dir / f"crolli_{year}.txt"
        if volume_file.exists():
            try:
                collapse_volume_df = read_collapse_volume_file(volume_file)
            except Exception as e:
                logger.error(f"Error reading volume file {volume_file}: {e}")
                raise e
            has_volume_file = True
        else:
            has_volume_file = False
        if not has_volume_file:
            logger.warning(
                f"No volume file found for year {year}, proceeding without volume data."
            )

        if not files:
            logger.warning(f"No shapefiles found in {year_dir}, skipping...")
            continue

        for file in tqdm(files, desc="Processing shapefiles"):
            logger.debug(f"Reading {file}...")

            # Skip shapefile with _vol in the name (they don't have geometry)
            if "_vol" in file.name:
                logger.info(f"Skipping {file} (volume shapefile)")
                continue

            # Extract date from filename
            try:
                date_str = file.stem.split("-")[0]
                date = pd.to_datetime(date_str, format="%Y%m%d").date()
            except Exception as e:
                logger.error(f"Error parsing date from {file}: {e}")
                raise ValueError(
                    f"Filename {file.name} does not match expected format."
                ) from e
                # continue

            # Fetch image IDs for the given date, ordered by acquisition time descending
            try:
                image_ids = fetch_image_ids(
                    db_engine,
                    date=date.isoformat(),
                    order_by="acquisition_timestamp DESC",
                )
                images_metadata = fetch_image_metadata_by_ids(
                    db_engine=db_engine, image_id=image_ids
                )
            except Exception as e:
                logger.error(f"Error fetching image IDs for {file}: {e}")
                continue

            # Pick up center image in the list
            idx = len(image_ids) // 2
            image_id = image_ids[idx]
            logger.debug(
                f"Selected image ID: {image_id} - {images_metadata.iloc[idx].acquisition_timestamp}"
            )

            # Read geometry from file
            try:
                geom = read_shapely_geom_from_file(file, invert_y=True)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue

            # Get volume from collapse_volume_df if available
            volume = None
            if has_volume_file:
                match = collapse_volume_df.loc[
                    (collapse_volume_df["date"].dt.date == date)
                ]
                if not match.empty:
                    volume = float(match.iloc[0]["volume"])
                    logger.debug(f"Found volume {volume} for date {date}")
                else:
                    logger.debug(f"No volume entry found for date {date}")

            # Insert geometry into DB and get id
            area = np.round(geom.area, 5)
            collapse_id = write_geom_wkt_to_db(
                db_engine=db_engine,
                image_id=image_id,
                geom=geom,
                area=area,
                volume=volume,
            )
            logger.debug(f"Inserted collapse id: {collapse_id}")

    # # TEST FETCHING DATA AND PLOTTING
    # date = "2018-09-19"

    # query = """
    #     SELECT c.id, c.image_id, ST_AsText(c.geom) AS geom_wtk, area, volume
    #     FROM ppcx_app_collapse c
    #     JOIN ppcx_app_image img ON c.image_id = img.id
    #     WHERE img.acquisition_timestamp::date = %s
    # """
    # df_read = pd.read_sql(query, db_engine, params=(date,))
    # img = get_image(image_id=df_read["image_id"].values[0], config=config)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(img)
    # colormap = plt.get_cmap("tab10")
    # for i, g in enumerate(df_read["geom_wtk"]):
    #     g = shapely_wkt.loads(g)
    #     xs, ys = g.exterior.xy
    #     ax.fill(xs, ys, facecolor="none", edgecolor=colormap(i), linewidth=2)
    # ax.set_axis_off()
    # plt.show()
