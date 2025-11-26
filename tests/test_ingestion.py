import os
import sys
sys.path.insert(0, '../src')
import pandas as pd
from src.data_ingestion import DataIngestion


def test_sales_ingestion_loads_file():
    """Prueba básica: el archivo de ventas se carga y devuelve un DataFrame"""

    ingestor = DataIngestion(config_path="config/pipeline_config.yaml")
    data = ingestor.ingest_sales_data()

    # Debe ser un DataFrame
    assert isinstance(data, pd.DataFrame)

    # Debe contener estas columnas, aunque haya variaciones
    expected_cols = {"product_name", "quantity", "unit_price", "date"}
    assert expected_cols.issubset(set(data.columns))


def test_ingestion_returns_dict():
    """Valida que ingest_all() devuelve un diccionario con claves esperadas."""

    ingestor = DataIngestion(config_path="config/pipeline_config.yaml")
    data_dict = ingestor.ingest_all()

    assert isinstance(data_dict, dict)
    assert "sales" in data_dict
    assert "instagram" in data_dict
    assert "tiktok" in data_dict
    assert "youtube" in data_dict


def test_ingestion_sales_row_count():
    """Asegura que se cargan registros (no vacío)"""
    ingestor = DataIngestion(config_path="config/pipeline_config.yaml")
    df = ingestor.ingest_sales_data()

    assert len(df) > 0
