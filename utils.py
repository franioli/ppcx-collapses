import matplotlib
import pandas as pd
from sqlalchemy import create_engine

from ppcollapse.utils.config import ConfigManager

matplotlib.use("Qt5Agg")

config = ConfigManager(config_path="config.yaml")
db_engine = create_engine(
    "postgresql://postgres:postgresppcx@150.145.51.193:5434/sandbox"
)


def get_collapses_df(db_engine) -> pd.DataFrame:
    """Read all collapse records from DB and return dataframe with parsed dates."""
    query = """
        SELECT c.id, img.acquisition_timestamp::date AS date,
               c.image_id, ST_AsText(c.geom) AS geom_wkt,
               c.area, c.volume
        FROM ppcx_app_collapse c
        JOIN ppcx_app_image img ON c.image_id = img.id
        ORDER BY img.acquisition_timestamp DESC, c.id ASC
    """
    df = pd.read_sql(query, db_engine, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_collapses_by_date(engine, date: str, limit: int | None = None) -> pd.DataFrame:
    """Return collapse rows. If date provided, filter by image acquisition date (date string YYYY-MM-DD)."""
    query = """
        SELECT c.id, img.acquisition_timestamp::date AS date,
            c.image_id, ST_AsText(c.geom) AS geom_wkt,
            c.area, c.volume
        FROM ppcx_app_collapse c
        JOIN ppcx_app_image img ON c.image_id = img.id
        WHERE img.acquisition_timestamp::date = %s
        ORDER BY img.acquisition_timestamp DESC, c.id ASC
    """
    if limit is not None:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, engine, params=(date,))
    return df


if __name__ == "__main__":
    pass
