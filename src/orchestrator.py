"""
Pipeline Orchestrator - MEJORADO
Orquesta la ejecucion completa del pipeline de datos
Incluye: Ingesta, Transformacion, Validacion, Segmentacion y Correlacion
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from data_ingestion import DataIngestion
from data_validation import DataValidator
from data_transformation import DataTransformation
from customer_segmentation import CustomerSegmentation
from correlation_analysis import CorrelationAnalysis

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
    """Orquestador principal del pipeline DataOps con 5 etapas"""

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
        
        # Crear instancias de modulos
        try:
            self.ingestor = DataIngestion(config_path)
            self.validator = DataValidator()
            self.transformer = DataTransformation()
            self.segmentation = CustomerSegmentation()
            self.correlation = CorrelationAnalysis()
            logger.info("Modulos del pipeline inicializados correctamente")
        except Exception as e:
            logger.error(f"Error inicializando modulos: {e}")
            raise

    def run(self):
        """Ejecuta el pipeline completo (5 etapas)"""
        self.execution_stats['start_time'] = datetime.now()
        self.execution_stats['status'] = 'running'
        
        logger.info("\n" + "="*35)
        logger.info("INICIANDO EJECUCION DEL PIPELINE COMPLETO")
        logger.info("="*35 + "\n")
        
        try:
            # ========================================
            # ETAPA 1: INGESTA
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("ETAPA 1/5: INGESTA DE DATOS")
            logger.info("="*70)
            
            raw_data = self._run_ingestion()
            self.execution_stats['stages_completed'].append('ingestion')
            
            # ========================================
            # ETAPA 2: TRANSFORMACION
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("ETAPA 2/5: TRANSFORMACION DE DATOS")
            logger.info("="*70)
            
            transformed_data = self._run_transformation(raw_data)
            self.execution_stats['stages_completed'].append('transformation')
            
            # ========================================
            # ETAPA 3: VALIDACION
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("ETAPA 3/5: VALIDACION DE CALIDAD")
            logger.info("="*70)
            
            validation_results = self._run_validation(transformed_data)
            self.execution_stats['stages_completed'].append('validation')
            
            # ========================================
            # ETAPA 4: SEGMENTACION (RFM + K-means + PCA)
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("ETAPA 4/5: SEGMENTACION DE CLIENTES")
            logger.info("="*70)
            
            segmentation_results = self._run_segmentation(transformed_data)
            self.execution_stats['stages_completed'].append('segmentation')
            
            # ========================================
            # ETAPA 5: ANALISIS DE CORRELACION
            # ========================================
            logger.info("\n" + "="*70)
            logger.info("ETAPA 5/5: ANALISIS DE CORRELACION")
            logger.info("="*70)
            
            correlation_results = self._run_correlation(
                transformed_data, 
                segmentation_results
            )
            self.execution_stats['stages_completed'].append('correlation')
            
            # ========================================
            # FINALIZACION
            # ========================================
            self.execution_stats['status'] = 'completed'
            self._finalize_execution(validation_results)
            
            return {
                'data': transformed_data,
                'validation': validation_results,
                'segmentation': segmentation_results,
                'correlation': correlation_results,
                'stats': self.execution_stats
            }
            
        except Exception as e:
            self.execution_stats['status'] = 'failed'
            self.execution_stats['errors'].append({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            logger.error("\n" + "="*35)
            logger.error("ERROR EN LA EJECUCION DEL PIPELINE")
            logger.error("="*35)
            logger.error(f"Error: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            return None

    def _run_ingestion(self):
        """Ejecuta la etapa de ingesta"""
        try:
            logger.info("Iniciando ingesta de datos...")
            
            raw_data = self.ingestor.ingest_all()
            
            # Resumen
            logger.info("\nResumen de ingesta:")
            for source, df in raw_data.items():
                if df is not None and hasattr(df, '__len__'):
                    logger.info(f"  {source}: {len(df)} registros")
                else:
                    logger.warning(f"  {source}: No disponible")
            
            logger.info("\nIngesta completada exitosamente")
            return raw_data
            
        except Exception as e:
            logger.error(f"Error en ingesta: {e}")
            self.execution_stats['stages_failed'].append('ingestion')
            raise

    def _run_transformation(self, raw_data):
        """Ejecuta la etapa de transformacion"""
        try:
            logger.info("Iniciando transformacion de datos...")
            
            transformed_data = self.transformer.transform_all(raw_data)
            
            # Resumen
            logger.info("\nResumen de transformacion:")
            for name, df in transformed_data.items():
                if df is not None and hasattr(df, '__len__'):
                    logger.info(f"  {name}: {len(df)} registros")
            
            # Verificar archivos criticos
            critical_files = ['sales', 'social_media_merged', 'daily_aggregates']
            missing = [f for f in critical_files if f not in transformed_data or transformed_data[f] is None]
            
            if missing:
                logger.warning(f"Archivos criticos faltantes: {missing}")
            else:
                logger.info("Todos los archivos criticos generados")
            
            logger.info("\nTransformacion completada exitosamente")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error en transformacion: {e}")
            self.execution_stats['stages_failed'].append('transformation')
            raise

    def _run_validation(self, transformed_data):
        """Ejecuta la etapa de validacion"""
        try:
            logger.info("Iniciando validacion de datos transformados...")
            
            validation_results = self.validator.validate_all(transformed_data)
            
            # Analisis de resultados
            passed = sum(1 for v in validation_results.values() if v)
            total = len(validation_results)
            success_rate = (passed / total * 100) if total > 0 else 0
            
            logger.info(f"\nResultados de validacion:")
            logger.info(f"  Datasets validados: {passed}/{total} ({success_rate:.1f}%)")
            
            for dataset, result in validation_results.items():
                status = "PASO" if result else "ADVERTENCIAS"
                logger.info(f"  {status}: {dataset}")
            
            # Decision sobre continuar o no
            if not all(validation_results.values()):
                logger.warning("\nADVERTENCIA: Algunos datasets tienen problemas de calidad")
                logger.warning("   El pipeline continuo, pero revisa los logs para mas detalles")
            else:
                logger.info("\nTodos los datasets pasaron la validacion")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error en validacion: {e}")
            self.execution_stats['stages_failed'].append('validation')
            raise

    def _run_segmentation(self, transformed_data):
        """
        Ejecuta la etapa de segmentacion de clientes
        """
        try:
            logger.info("Iniciando segmentacion de clientes...")
            
            # Verificar datos necesarios
            if 'sales' not in transformed_data or transformed_data['sales'] is None:
                raise ValueError("No hay datos de ventas para segmentar")
            
            sales_df = transformed_data['sales']
            
            # Ejecutar segmentacion (RFM + K-means + PCA)
            segmentation_results = self.segmentation.segment_customers(
                sales_df, 
                n_clusters=4
            )
            
            # Generar visualizaciones
            logger.info("\nGenerando visualizaciones de segmentacion...")
            self.segmentation.plot_segments(
                segmentation_results['segments'],
                segmentation_results['pca_coords'],
                save_path='dashboards/static'
            )
            
            # Resumen
            logger.info("\nResumen de segmentacion:")
            logger.info(f"  Clientes segmentados: {len(segmentation_results['segments'])}")
            logger.info(f"  Segmentos identificados: {segmentation_results['segments']['segment_name'].nunique()}")
            
            segment_dist = segmentation_results['segments']['segment_name'].value_counts()
            for seg_name, count in segment_dist.items():
                pct = (count / len(segmentation_results['segments'])) * 100
                logger.info(f"    {seg_name}: {count} clientes ({pct:.1f}%)")
            
            logger.info("\nSegmentacion completada exitosamente")
            return segmentation_results
            
        except Exception as e:
            logger.error(f"Error en segmentacion: {e}")
            self.execution_stats['stages_failed'].append('segmentation')
            raise

    def _run_correlation(self, transformed_data, segmentation_results):
        """
        Ejecuta la etapa de analisis de correlacion
        """
        try:
            logger.info("Iniciando analisis de correlacion...")
            
            # Verificar datos necesarios
            required_data = ['sales', 'daily_aggregates']
            missing = [d for d in required_data if d not in transformed_data or transformed_data[d] is None]
            
            if missing:
                raise ValueError(f"Faltan datos necesarios: {missing}")
            
            sales_df = transformed_data['sales']
            daily_agg_df = transformed_data['daily_aggregates']
            segments_df = segmentation_results['segments']
            
            # Ejecutar analisis de correlacion
            correlation_results = self.correlation.analyze_all(
                daily_agg_df,
                segments_df,
                sales_df
            )
            
            # Generar visualizaciones
            logger.info("\nGenerando visualizaciones de correlacion...")
            self.correlation.plot_correlations(
                daily_agg_df,
                correlation_results.get('lag'),
                correlation_results.get('by_segment'),
                save_path='dashboards/static'
            )
            
            # Resumen
            logger.info("\nResumen de correlacion:")
            
            if correlation_results.get('general'):
                gen = correlation_results['general']
                r_value = gen['pearson']['r']
                p_value = gen['pearson']['p_value']
                sig = "Significativa" if gen['pearson']['significant'] else "No significativa"
                
                logger.info(f"  Correlacion general (Pearson):")
                logger.info(f"    r = {r_value:.3f}")
                logger.info(f"    p-value = {p_value:.4f}")
                logger.info(f"    {sig}")
            
            if correlation_results.get('lag') is not None and not correlation_results['lag'].empty:
                best_lag = correlation_results['lag'].loc[
                    correlation_results['lag']['correlation'].abs().idxmax()
                ]
                logger.info(f"  Lag optimo: {best_lag['lag_days']:.0f} dias (r={best_lag['correlation']:.3f})")
            
            if correlation_results.get('by_segment') is not None and not correlation_results['by_segment'].empty:
                logger.info(f"  Segmentos analizados: {len(correlation_results['by_segment'])}")
            
            if correlation_results.get('by_product') is not None and not correlation_results['by_product'].empty:
                logger.info(f"  Productos analizados: {len(correlation_results['by_product'])}")
            
            logger.info("\nAnalisis de correlacion completado exitosamente")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error en analisis de correlacion: {e}")
            self.execution_stats['stages_failed'].append('correlation')
            raise

    def _finalize_execution(self, validation_results):
        """Finaliza la ejecucion y genera resumen"""
        self.execution_stats['end_time'] = datetime.now()
        duration = self.execution_stats['end_time'] - self.execution_stats['start_time']
        self.execution_stats['duration_seconds'] = duration.total_seconds()
        
        logger.info("\n" + "="*35)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("="*35)
        
        logger.info(f"\nInformacion de ejecucion:")
        logger.info(f"  Inicio: {self.execution_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Fin: {self.execution_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Duracion: {self.execution_stats['duration_seconds']:.2f} segundos")
        
        logger.info(f"\nEtapas completadas ({len(self.execution_stats['stages_completed'])}/5):")
        for stage in self.execution_stats['stages_completed']:
            logger.info(f"  {stage.capitalize()}")
        
        if self.execution_stats['stages_failed']:
            logger.warning(f"\nEtapas con problemas:")
            for stage in self.execution_stats['stages_failed']:
                logger.warning(f"  {stage}")
        
        # Verificar archivos generados
        logger.info(f"\nArchivos generados:")
        processed_path = Path('data/processed')
        if processed_path.exists():
            files = list(processed_path.glob('*.csv'))
            for file in sorted(files):
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  {file.name} ({size_mb:.2f} MB)")
        
        # Verificar visualizaciones generadas
        logger.info(f"\nVisualizaciones generadas:")
        static_path = Path('dashboards/static')
        if static_path.exists():
            images = list(static_path.glob('*.png'))
            for img in sorted(images):
                logger.info(f"  {img.name}")
        
        # Validacion general
        all_valid = all(validation_results.values())
        if all_valid:
            logger.info(f"\nCalidad de datos: TODOS LOS CHECKS PASADOS")
        else:
            logger.warning(f"\nCalidad de datos: REVISAR ADVERTENCIAS")
        
        logger.info("\n" + "="*70)
        logger.info("Log completo guardado en: pipeline_execution.log")
        logger.info("="*70 + "\n")
    
    def get_execution_stats(self):
        """Retorna estadisticas de ejecucion"""
        return self.execution_stats


def main():
    """Funcion principal"""
    print("\n" + "="*70)
    print("SUPPLEMENT SALES PIPELINE - DATAOPS PROJECT")
    print("Pipeline Completo: Ingesta -> Transformacion -> Validacion")
    print("                  -> Segmentacion -> Correlacion")
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
            print(f"\nEstadisticas de ejecucion:")
            print(f"  Duracion total: {stats['duration_seconds']:.2f} segundos")
            print(f"  Etapas completadas: {len(stats['stages_completed'])}/5")
            print(f"    - {', '.join(stats['stages_completed'])}")
            
            print(f"\nResultados guardados:")
            print(f"  Datos procesados: data/processed/")
            print(f"  Visualizaciones: dashboards/static/")
            print(f"  Log detallado: pipeline_execution.log")
            
            print("\nProximos pasos:")
            print("  1. Revisar visualizaciones en dashboards/static/")
            print("  2. Ejecutar dashboard: streamlit run src/dash_results.py")
            print("  3. Revisar correlaciones y segmentos para estrategias de marketing")
            
            return 0
        else:
            print("\n" + "="*70)
            print("PIPELINE FALLO")
            print("="*70)
            print("\nRevisa el archivo pipeline_execution.log para mas detalles")
            return 1
            
    except Exception as e:
        print(f"\nError fatal: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())