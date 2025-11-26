import subprocess
import sys
from pathlib import Path

def test_pipeline_integration():
    """
    Prueba de integración:
    - Ejecuta el pipeline completo
    - Verifica que termine correctamente
    - Confirma presencia de outputs y logs
    """

    # Ejecutar el pipeline a través del archivo del orquestador
    result = subprocess.run(
        [sys.executable, "src/orchestrator.py"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, (
        f"Pipeline execution failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert "PIPELINE EJECUTADO EXITOSAMENTE" in result.stdout, (
        "El pipeline no mostró el mensaje esperado de ejecución exitosa."
    )
    log_file = Path("pipeline_execution.log")
    assert log_file.exists(), "El archivo pipeline_execution.log no fue generado."

    processed_path = Path("data/processed")
    assert processed_path.exists(), "La carpeta data/processed no existe."

    csv_files = list(processed_path.glob("*.csv"))
    assert len(csv_files) > 0, "No se generaron archivos CSV en data/processed."

    sizes = [f.stat().st_size for f in csv_files]
    assert any(size > 10 for size in sizes), "Los archivos en data/processed están vacíos."

