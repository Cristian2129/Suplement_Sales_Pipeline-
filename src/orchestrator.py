"""
Pipeline Orchestrator
Orquesta la ejecución completa del pipeline de datos
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from data_ingestion import DataIngestion
from data_validation import DataValidator
from data_transformation import DataTransformation

# Configurar logging con archivo
def setup_logging():
    """Configura logging a consola y archivo"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Crear handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler('pipeline_execution.log')
    file_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()


class PipelineOrchestrator:
    """Orquestador principal del pipeline DataOps"""

    def __init__(self, config_path="config/pipeline_config.yaml"):
        """Inicializa el orquestador"""
        logger.info("Inicializando Pipeline Orchestrator...")
        
        self.config_path = config_path
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'status': 'initialized',
            'stages_completed': [],
            'stages_failed': [],
            'errors': []
        }
        
        # Crear instancias de módulos
        try:
            self.ingestor = DataIngestion(config_path)
            self.validator = DataValidator()
            self.transformer = DataTransformation()
            logger.info("Módulos del pipeline inicializados correctamente")
        except Exception as e:
            logger.error(f" Error inicializando módulos: {e}")
            raise

    def run(self):
        """Ejecuta el pipeline completo"""
        self.execution_stats['start_time'] = datetime.now()
        self.execution_stats['status'] = 'running'
        
        logger.info("\n" + ""*35)
        logger.info("INICIANDO EJECUCIÓN DEL PIPELINE")
        logger.info(" "*35 + "\n")
        
        try:
            # ========================================
            # STAGE 1: INGESTA
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("STAGE 1: INGESTA DE DATOS")
            logger.info("="*70)
            
            raw_data = self._run_ingestion()
            self.execution_stats['stages_completed'].append('ingestion')
            
            # ========================================
            # STAGE 2: TRANSFORMACIÓN
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("STAGE 2: TRANSFORMACIÓN DE DATOS")
            logger.info("="*70)
            
            transformed_data = self._run_transformation(raw_data)
            self.execution_stats['stages_completed'].append('transformation')
            
            # ========================================
            # STAGE 3: VALIDACIÓN (de datos transformados)
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("STAGE 3: VALIDACIÓN DE CALIDAD")
            logger.info("="*70)
            
            validation_results = self._run_validation(transformed_data)
            self.execution_stats['stages_completed'].append('validation')
            
            # ========================================
            # FINALIZACIÓN
            # ========================================
            self.execution_stats['status'] = 'completed'
            self._finalize_execution(validation_results)
            
            return {
                'data': transformed_data,
                'validation': validation_results,
                'stats': self.execution_stats
            }
            
        except Exception as e:
            self.execution_stats['status'] = 'failed'
            self.execution_stats['errors'].append({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            logger.error("\n" + " "*35)
            logger.error("ERROR EN LA EJECUCIÓN DEL PIPELINE")
            logger.error(" "*35)
            logger.error(f"Error: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            return None

    def _run_ingestion(self):
        """Ejecuta la etapa de ingesta"""
        try:
            logger.info("Iniciando ingesta de datos...")
            
            raw_data = self.ingestor.ingest_all()
            
            # Resumen
            logger.info("\n Resumen de ingesta:")
            for source, df in raw_data.items():
                if df is not None and hasattr(df, '__len__'):
                    logger.info(f" {source}: {len(df)} registros")
                else:
                    logger.warning(f" {source}: No disponible")
            
            logger.info("\n Ingesta completada exitosamente")
            return raw_data
            
        except Exception as e:
            logger.error(f" Error en ingesta: {e}")
            self.execution_stats['stages_failed'].append('ingestion')
            raise

    def _run_transformation(self, raw_data):
        """Ejecuta la etapa de transformación"""
        try:
            logger.info("Iniciando transformación de datos...")
            
            transformed_data = self.transformer.transform_all(raw_data)
            
            # Resumen
            logger.info("\n Resumen de transformación:")
            for name, df in transformed_data.items():
                if df is not None and hasattr(df, '__len__'):
                    logger.info(f"  {name}: {len(df)} registros")
            
            # Verificar archivos críticos
            critical_files = ['sales', 'social_media_merged', 'daily_aggregates']
            missing = [f for f in critical_files if f not in transformed_data or transformed_data[f] is None]
            
            if missing:
                logger.warning(f" Archivos críticos faltantes: {missing}")
            else:
                logger.info("Todos los archivos críticos generados")
            
            logger.info("\n Transformación completada exitosamente")
            return transformed_data
            
        except Exception as e:
            logger.error(f" Error en transformación: {e}")
            self.execution_stats['stages_failed'].append('transformation')
            raise

    def _run_validation(self, transformed_data):
        """Ejecuta la etapa de validación"""
        try:
            logger.info("Iniciando validación de datos transformados...")
            
            validation_results = self.validator.validate_all(transformed_data)
            
            # Análisis de resultados
            passed = sum(1 for v in validation_results.values() if v)
            total = len(validation_results)
            success_rate = (passed / total * 100) if total > 0 else 0
            
            logger.info(f"\n Resultados de validación:")
            logger.info(f"  Datasets validados: {passed}/{total} ({success_rate:.1f}%)")
            
            for dataset, result in validation_results.items():
                status = " PASÓ" if result else " ADVERTENCIAS"
                logger.info(f"  {dataset}: {status}")
            
            # Decisión sobre continuar o no
            if not all(validation_results.values()):
                logger.warning("\n  ADVERTENCIA: Algunos datasets tienen problemas de calidad")
                logger.warning("El pipeline continuó, pero revisa los logs para más detalles")
            else:
                logger.info("\n Todos los datasets pasaron la validación")
            
            return validation_results
            
        except Exception as e:
            logger.error(f" Error en validación: {e}")
            self.execution_stats['stages_failed'].append('validation')
            raise

    def _finalize_execution(self, validation_results):
        """Finaliza la ejecución y genera resumen"""
        self.execution_stats['end_time'] = datetime.now()
        duration = self.execution_stats['end_time'] - self.execution_stats['start_time']
        self.execution_stats['duration_seconds'] = duration.total_seconds()
        
        logger.info("\n" + " "*35)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(" "*35)
        
        logger.info(f"\n Información de ejecución:")
        logger.info(f"  Inicio: {self.execution_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Fin: {self.execution_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Duración: {self.execution_stats['duration_seconds']:.2f} segundos")
        
        logger.info(f"\n Etapas completadas:")
        for stage in self.execution_stats['stages_completed']:
            logger.info(f" {stage}")
        
        if self.execution_stats['stages_failed']:
            logger.warning(f"\n  Etapas con problemas:")
            for stage in self.execution_stats['stages_failed']:
                logger.warning(f"{stage}")
        
        # Verificar archivos generados
        logger.info(f"\n Archivos generados:")
        processed_path = Path('data/processed')
        if processed_path.exists():
            files = list(processed_path.glob('*.csv'))
            for file in sorted(files):
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f" {file.name} ({size_mb:.2f} MB)")
        
        # Validación general
        all_valid = all(validation_results.values())
        if all_valid:
            logger.info(f"\n Calidad de datos: TODOS LOS CHECKS PASADOS")
        else:
            logger.warning(f"\n  Calidad de datos: REVISAR ADVERTENCIAS")
        
        logger.info("\n" + "="*70)
        logger.info("Log completo guardado en: pipeline_execution.log")
        logger.info("="*70 + "\n")
    
    def get_execution_stats(self):
        """Retorna estadísticas de ejecución"""
        return self.execution_stats


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("SUPPLEMENT SALES PIPELINE - DATAOPS PROJECT")
    print("Segmentación de Clientes y Análisis de Redes Sociales")
    print("="*70 + "\n")
    
    try:
        # Crear y ejecutar orquestador
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        
        if result is not None:
            print("\n" + "="*70)
            print("PIPELINE EJECUTADO EXITOSAMENTE")
            print("="*70)
            
            stats = orchestrator.get_execution_stats()
            print(f"\n  Duración total: {stats['duration_seconds']:.2f} segundos")
            print(f" Etapas completadas: {', '.join(stats['stages_completed'])}")
            print(f" Datos procesados guardados en: data/processed/")
            print(f" Log detallado en: pipeline_execution.log")
            
            return 0
        else:
            print("\n" + "="*70)
            print(" PIPELINE FALLÓ")
            print("="*70)
            print("\n  Revisa el archivo pipeline_execution.log para más detalles")
            return 1
            
    except Exception as e:
        print(f"\n Error fatal: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())