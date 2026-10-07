"""The hand-written migration must create exactly what app/models.py describes."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.models import Base

BACKEND = Path(__file__).resolve().parents[1]


def offline_sql(capsys, *args) -> str:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", "postgresql+psycopg2://user:pass@localhost/kumbhmela")
    capsys.readouterr()
    getattr(command, args[0])(config, *args[1:], sql=True)
    return capsys.readouterr().out


def test_upgrade_creates_every_model_table_and_column(capsys):
    sql = offline_sql(capsys, "upgrade", "head")
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in sql
    assert "CREATE TYPE risk_level AS ENUM ('GREEN', 'YELLOW', 'ORANGE', 'RED')" in sql
    assert "geometry(POLYGON,4326)" in sql and "USING gist" in sql
    for table in Base.metadata.sorted_tables:
        create = sql.split(f"CREATE TABLE {table.name} (", 1)[1].split(");", 1)[0]
        for column in table.columns:
            assert f"\n    {column.name} " in create, f"{table.name}.{column.name} missing from migration"


def test_downgrade_drops_everything(capsys):
    sql = offline_sql(capsys, "downgrade", "001_initial:base")
    for table in ("alert_logs", "incidents", "zone_readings", "zones"):
        assert f"DROP TABLE {table}" in sql
    assert "DROP TYPE risk_level" in sql
