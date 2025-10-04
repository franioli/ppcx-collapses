DROP TABLE IF EXISTS ppcx_app_collapse;
CREATE TABLE IF NOT EXISTS ppcx_app_collapse (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES ppcx_app_image(id),
    geom geometry(Geometry,0),
    area DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);