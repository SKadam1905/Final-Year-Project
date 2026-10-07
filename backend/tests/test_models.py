from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.models import AlertLog, Base, Incident, Zone, ZoneReading


def ddl(model) -> str:
    return str(CreateTable(model.__table__).compile(dialect=postgresql.dialect()))


def test_schema_compiles_for_postgis():
    assert set(Base.metadata.tables) == {"zones", "zone_readings", "incidents", "alert_logs"}
    assert "geometry(POLYGON,4326)" in ddl(Zone)
    assert "risk_level" in ddl(ZoneReading)
    assert "REFERENCES zones (id)" in ddl(Incident)
    assert "REFERENCES incidents (id)" in ddl(AlertLog)
