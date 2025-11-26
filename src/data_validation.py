"""
Data Validation Module
Valida calidad de datos después de la transformación
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Clase para validar calidad de datos transformados"""
    
    def __init__(self):
        """Inicializa el validador"""
        self.validation_results = {}
        self.all_passed = True
        
    # =========================================================
    # VALIDACIONES GENERALES (reutilizables)
    # =========================================================
    
    def _validate_required_columns(self, df, required_cols, dataset_name):
        """Verifica que existan columnas requeridas"""
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            logger.error(f"[{dataset_name}] Columnas faltantes: {missing}")
            return False
        
        logger.info(f"[{dataset_name}] Todas las columnas requeridas presentes")
        return True
    
    def _validate_nulls(self, df, critical_cols, dataset_name):
        """Verifica nulos en columnas criticas"""
        # Filtrar solo columnas que existen
        existing_cols = [col for col in critical_cols if col in df.columns]
        
        if not existing_cols:
            logger.warning(f"[{dataset_name}] Ninguna columna critica encontrada")
            return False
        
        nulls = df[existing_cols].isnull().sum()
        nulls = nulls[nulls > 0]
        
        if len(nulls) > 0:
            logger.warning(f"[{dataset_name}] Nulos encontrados:\n{nulls}")
            return False
        
        logger.info(f"[{dataset_name}] Sin nulos en columnas criticas")
        return True
    
    def _validate_duplicates(self, df, key_cols, dataset_name):
        """Verifica duplicados"""
        if not key_cols:
            return True
        
        # Filtrar solo columnas que existen
        existing_cols = [col for col in key_cols if col in df.columns]
        
        if not existing_cols:
            logger.warning(f"[{dataset_name}] Ninguna columna de clave encontrada para verificar duplicados")
            return True  # No es crítico si no hay columnas para verificar
        
        dups = df.duplicated(subset=existing_cols).sum()
        if dups > 0:
            logger.warning(f"[{dataset_name}] {dups} duplicados detectados")
            return False
        
        logger.info(f"[{dataset_name}] Sin duplicados")
        return True
    
    def _validate_ranges(self, df, numerical_cols, dataset_name, min_val=0):
        """Verifica rangos válidos en columnas numéricas"""
        issues = []
        
        for col in numerical_cols:
            if col in df.columns:
                negatives = (df[col] < min_val).sum()
                if negatives > 0:
                    issues.append(f"{col}: {negatives} valores < {min_val}")
        
        if issues:
            logger.warning(f"[{dataset_name}] Valores fuera de rango: {', '.join(issues)}")
            return False
        
        logger.info(f"[{dataset_name}] Rangos numéricos válidos")
        return True
    
    def _validate_dates(self, df, date_col, dataset_name):
        """Valida fechas razonables"""
        if date_col not in df.columns:
            logger.warning(f"[{dataset_name}] Columna {date_col} no encontrada")
            return True # No es crítico si no existe
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            logger.warning(f"[{dataset_name}] {date_col} no es datetime")
            return False
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        if min_date.year < 2010 or max_date.year > 2030:
            logger.warning(
                f"[{dataset_name}] Fechas fuera de rango esperado: "
                f"{min_date.date()} - {max_date.date()}"
            )
            return False
        
        logger.info(f"[{dataset_name}] Fechas válidas ({min_date.date()} a {max_date.date()})")
        return True
    
    def _validate_data_freshness(self, df, date_col, dataset_name, max_days_old=365):
        """Verifica que los datos no sean muy antiguos"""
        if date_col not in df.columns:
            return True
        
        latest_date = df[date_col].max()
        current_date = pd.Timestamp.now()
        days_old = (current_date - latest_date).days
        
        if days_old > max_days_old:
            logger.warning(
                f"[{dataset_name}] Datos antiguos: última fecha hace {days_old} días"
            )
            return False
        
        logger.info(f"[{dataset_name}] Datos frescos (última fecha hace {days_old} días)")
        return True
    
    def _validate_unique_count(self, df, col, dataset_name, min_unique=1):
        """Verifica que haya suficiente variedad en una columna"""
        if col not in df.columns:
            logger.warning(f"[{dataset_name}] Columna '{col}' no encontrada para validar unicidad")
            return True  # No es crítico si la columna no existe
        
        unique_count = df[col].nunique()
        
        if unique_count < min_unique:
            logger.warning(
                f"[{dataset_name}] Poca variedad en {col}: {unique_count} valores unicos"
            )
            return False
        
        logger.info(f"[{dataset_name}] {col}: {unique_count} valores unicos")
        return True
        
    # =========================================================
    # VALIDACIONES ESPECÍFICAS POR DATASET
    # =========================================================
    
    def validate_sales(self, df):
        """Valida datos de ventas"""
        logger.info("\n" + "="*70)
        logger.info("VALIDANDO DATOS DE VENTAS")
        logger.info("="*70)
        
        if df is None or df.empty:
            logger.error("[SALES] Dataset vacío o None")
            return False
        
        dataset = "SALES"
        
        results = {
            'required_columns': self._validate_required_columns(
                df, ['customer_id', 'product_id', 'date', 'amount'], dataset
            ),
            'nulls': self._validate_nulls(
                df, ['customer_id', 'product_id', 'date'], dataset
            ),
            'duplicates': self._validate_duplicates(
                df, ['transaction_id'], dataset
            ),
            'ranges': self._validate_ranges(
                df, ['quantity', 'unit_price', 'amount'], dataset
            ),
            'dates': self._validate_dates(df, 'date', dataset),
            'unique_customers': self._validate_unique_count(
                df, 'customer_id', dataset, min_unique=10
            ),
            'unique_products': self._validate_unique_count(
                df, 'product_id', dataset, min_unique=5
            )
        }
        
        all_valid = all(results.values())
        
                # Resumen
        logger.info(f"\n[{dataset}] Resumen:")
        logger.info(f" Total registros: {len(df)}")

        # Solo mostrar métricas si las columnas existen
        if 'customer_id' in df.columns:
            logger.info(f" Clientes únicos: {df['customer_id'].nunique()}")
        else:
            logger.info(f" Clientes únicos: N/A (columna no existe)")

        if 'product_id' in df.columns:
            logger.info(f" Productos únicos: {df['product_id'].nunique()}")
        else:
            logger.info(f" Productos únicos: N/A (columna no existe)")

        logger.info(f" Checks pasados: {sum(results.values())}/{len(results)}")
        if all_valid:
            logger.info(f"[{dataset}] VALIDACIÓN EXITOSA")
        else:
            logger.warning(f"[{dataset}] VALIDACIÓN CON ADVERTENCIAS")
        
        logger.info("="*70)
        
        self.validation_results['sales'] = results
        return all_valid
    
    def validate_social_media(self, df, platform):
        """Valida datos de redes sociales individuales"""
        logger.info(f"\n[{platform.upper()}] Validando...")
        
        if df is None or df.empty:
            logger.warning(f"[{platform.upper()}] Dataset vacío")
            return True # No crítico si no hay datos
        
        dataset = platform.upper()
        
        results = {
            'required_columns': self._validate_required_columns(
                df, ['date', 'engagement_rate'], dataset
            ),
            'nulls': self._validate_nulls(df, ['date'], dataset),
            'ranges': self._validate_ranges(
                df, ['engagement_rate'], dataset, min_val=0
            ),
            'dates': self._validate_dates(df, 'date', dataset)
        }
        
        all_valid = all(results.values())
        
        if all_valid:
            logger.info(f"[{dataset}] Validación exitosa ({len(df)} registros)")
        else:
            logger.warning(f"[{dataset}] Validación con advertencias")
        
        self.validation_results[platform] = results
        return all_valid
    
    def validate_social_media_merged(self, df):
        """Valida datos de redes sociales combinados"""
        logger.info("\n" + "="*70)
        logger.info("VALIDANDO SOCIAL MEDIA MERGED")
        logger.info("="*70)
        
        if df is None or df.empty:
            logger.error("[SOCIAL_MERGED] Dataset vacío")
            return False
        
        dataset = "SOCIAL_MERGED"
        
        results = {
            'required_columns': self._validate_required_columns(
                df, ['date', 'platform', 'engagement_rate'], dataset
            ),
            'nulls': self._validate_nulls(df, ['date', 'platform'], dataset),
            'ranges': self._validate_ranges(
                df, ['engagement_rate'], dataset, min_val=0
            ),
            'dates': self._validate_dates(df, 'date', dataset),
            'platforms': self._validate_unique_count(
                df, 'platform', dataset, min_unique=1
            )
        }
        
        all_valid = all(results.values())
        
        # Resumen por plataforma
        logger.info(f"\n[{dataset}] Distribución por plataforma:")
        for platform, count in df['platform'].value_counts().items():
            logger.info(f" {platform}: {count} registros")
        
        logger.info(f"\n[{dataset}] Checks pasados: {sum(results.values())}/{len(results)}")
        
        if all_valid:
            logger.info(f"[{dataset}] VALIDACIÓN EXITOSA")
        else:
            logger.warning(f"[{dataset}] VALIDACIÓN CON ADVERTENCIAS")
        
        logger.info("="*70)
        
        self.validation_results['social_media_merged'] = results
        return all_valid
    
    def validate_daily_aggregates(self, df):
        """Valida agregados diarios"""
        logger.info("\n" + "="*70)
        logger.info("VALIDANDO DAILY AGGREGATES")
        logger.info("="*70)
        
        if df is None or df.empty:
            logger.error("[DAILY_AGG] Dataset vacío")
            return False
        
        dataset = "DAILY_AGG"
        
        results = {
            'required_columns': self._validate_required_columns(
                df, ['date', 'total_sales', 'num_transactions'], dataset
            ),
            'nulls': self._validate_nulls(df, ['date'], dataset),
            'ranges': self._validate_ranges(
                df, ['total_sales', 'num_transactions'], dataset, min_val=0
            ),
            'dates': self._validate_dates(df, 'date', dataset),
            'date_continuity': self._validate_date_continuity(df, 'date', dataset)
        }
        
        all_valid = all(results.values())
        
        # Resumen
        logger.info(f"\n[{dataset}] Resumen:")
        logger.info(f" Total días: {len(df)}")
        logger.info(f" Rango: {df['date'].min().date()} a {df['date'].max().date()}")
        logger.info(f" Ventas totales: ${df['total_sales'].sum():,.2f}")
        logger.info(f" Checks pasados: {sum(results.values())}/{len(results)}")
        
        if all_valid:
            logger.info(f"[{dataset}] VALIDACIÓN EXITOSA")
        else:
            logger.warning(f"[{dataset}] VALIDACIÓN CON ADVERTENCIAS")
        
        logger.info("="*70)
        
        self.validation_results['daily_aggregates'] = results
        return all_valid
    
    def _validate_date_continuity(self, df, date_col, dataset_name):
        """Verifica que no haya gaps grandes en fechas"""
        if date_col not in df.columns:
            return True
        
        df_sorted = df.sort_values(date_col)
        date_diffs = df_sorted[date_col].diff().dt.days
        
        max_gap = date_diffs.max()
        
        if max_gap > 30: # Gap de más de 30 días
            logger.warning(
                f"[{dataset_name}] Gap grande en fechas: {max_gap} días"
            )
            return False
        
        logger.info(f"[{dataset_name}] Continuidad de fechas OK (gap máximo: {max_gap} días)")
        return True
    
    # =========================================================
    # VALIDAR TODO
    # =========================================================
    
    def validate_all(self, data_dict):
        """Valida todos los datasets transformados"""
        logger.info("\n" + " "*20)
        logger.info("INICIANDO VALIDACIÓN COMPLETA DE DATOS")
        logger.info(" "*20 + "\n")
        
        validations = {}
        
        # Ventas
        if data_dict.get('sales') is not None:
            validations['sales'] = self.validate_sales(data_dict['sales'])
        
        # Redes sociales individuales
        for platform in ['instagram', 'tiktok', 'youtube']:
            if data_dict.get(platform) is not None:
                validations[platform] = self.validate_social_media(
                    data_dict[platform], platform
                )
        
        # Social media merged
        if data_dict.get('social_media_merged') is not None:
            validations['social_media_merged'] = self.validate_social_media_merged(
                data_dict['social_media_merged']
            )
        
        # Daily aggregates
        if data_dict.get('daily_aggregates') is not None:
            validations['daily_aggregates'] = self.validate_daily_aggregates(
                data_dict['daily_aggregates']
            )
        
        # Resultado general
        self.all_passed = all(validations.values())
        
        # Resumen final
        self._print_validation_summary(validations)
        
        # Generar reporte
        self._generate_validation_report(validations)
        
        return validations
    
    def _print_validation_summary(self, validations):
        """Imprime resumen de validación"""
        logger.info("\n" + "="*70)
        logger.info("RESUMEN DE VALIDACIÓN")
        logger.info("="*70)
        
        passed = sum(validations.values())
        total = len(validations)
        
        for dataset, result in validations.items():
            status = "PASÓ" if result else "ADVERTENCIAS"
            logger.info(f" {dataset.upper()}: {status}")
        
        logger.info(f"\nTotal: {passed}/{total} datasets validados exitosamente")
        
        if self.all_passed:
            logger.info("\nTODOS LOS DATASETS PASARON LA VALIDACIÓN")
        else:
            logger.warning("\nALGUNOS DATASETS TIENEN ADVERTENCIAS")
        
        logger.info("="*70 + "\n")
    
    def _generate_validation_report(self, validations):
        """Genera reporte de validación en JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'all_passed': self.all_passed,
            'summary': validations,
            'details': self.validation_results
        }
        
        # Guardar reporte
        report_path = Path('data/processed')
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f" Reporte de validación guardado: {report_file}")


if __name__ == '__main__':
    # Test con datos procesados
    logger.info("\n INICIANDO VALIDACIÓN DE DATOS PROCESADOS\n")
    
    # Leer datos procesados
    data = {}
    
    try:
        data['sales'] = pd.read_csv('data/processed/sales_processed.csv', parse_dates=['date'])
        logger.info(" sales_processed.csv cargado")
    except:
        logger.warning(" sales_processed.csv no encontrado")
    
    try:
        data['instagram'] = pd.read_csv('data/processed/instagram_processed.csv', parse_dates=['date'])
        logger.info(" instagram_processed.csv cargado")
    except:
        logger.warning(" instagram_processed.csv no encontrado")
    
    try:
        data['tiktok'] = pd.read_csv('data/processed/tiktok_processed.csv', parse_dates=['date'])
        logger.info(" tiktok_processed.csv cargado")
    except:
        logger.warning(" tiktok_processed.csv no encontrado")
    
    try:
        data['youtube'] = pd.read_csv('data/processed/youtube_processed.csv', parse_dates=['date'])
        logger.info(" youtube_processed.csv cargado")
    except:
        logger.warning(" youtube_processed.csv no encontrado")
    
    try:
        data['social_media_merged'] = pd.read_csv('data/processed/social_media_merged_processed.csv', parse_dates=['date'])
        logger.info(" social_media_merged_processed.csv cargado")
    except:
        logger.warning(" social_media_merged_processed.csv no encontrado")
    
    try:
        data['daily_aggregates'] = pd.read_csv('data/processed/daily_aggregates_processed.csv', parse_dates=['date'])
        logger.info(" daily_aggregates_processed.csv cargado")
    except:
        logger.warning(" daily_aggregates_processed.csv no encontrado")
    
    # Validar
    validator = DataValidator()
    results = validator.validate_all(data)
    
    print("\n" + "="*70)
    print("VALIDACIÓN COMPLETADA")
    print("="*70)