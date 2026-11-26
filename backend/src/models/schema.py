"""Bring an existing database up to the columns and index used by round versioning."""

from sqlalchemy import inspect, text

from src.models.database import engine


def upgrade():
    """Add round-versioning columns to a database created before this release.

    Fresh deployments get the full schema from ``metadata.create_all``; this only fills
    in what an older database is missing, so both paths end up with the same shape.
    """
    inspector = inspect(engine)
    if 'analysis_runs' not in inspector.get_table_names():
        return
    existing = {column['name'] for column in inspector.get_columns('analysis_runs')}
    additions = (
        ('round_key', 'VARCHAR(64)'),
        ('version', 'INTEGER NOT NULL DEFAULT 0'),
        ('active', 'INTEGER NOT NULL DEFAULT 0'),
    )
    with engine.begin() as connection:
        for name, definition in additions:
            if name not in existing:
                connection.execute(text(f'ALTER TABLE analysis_runs ADD COLUMN {name} {definition}'))
    names = {index['name'] for index in inspect(engine).get_indexes('analysis_runs')}
    if 'ux_analysis_runs_round_version' not in names:
        with engine.begin() as connection:
            connection.execute(text('CREATE UNIQUE INDEX ux_analysis_runs_round_version '
                                    'ON analysis_runs (dataset_id, round_key, version)'))
