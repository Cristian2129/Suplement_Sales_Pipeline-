"""
Data Transformation Module
Responsabilidad: Limpieza, enriquecimiento y transformación de datos
Aquí se crean IDs, se calculan métricas y se generan features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataTransformation:
    """Clase para transformar y limpiar datos"""
    
    def __init__(self):
        """Inicializa el transformador"""
        self.processed_data = {}
        
    def transform_all(self, data_dict):
        """
        Aplica todas las transformaciones disponibles.
        Recibe data_dict desde data_ingestion.ingest_all()
        """
        logger.info("=" * 70)
        logger.info("INICIANDO TRANSFORMACIÓN DE DATOS")
        logger.info("=" * 70)

        transformed = {}

        # 1. Transformar ventas
        if data_dict.get("sales") is not None:
            transformed["sales"] = self.transform_sales_data(data_dict["sales"])

        # 2. Transformar redes sociales individuales
        if data_dict.get("instagram") is not None:
            transformed["instagram"] = self.transform_social_media_data(
                data_dict["instagram"], 'instagram'
            )

        if data_dict.get("tiktok") is not None:
            transformed["tiktok"] = self.transform_social_media_data(
                data_dict["tiktok"], 'tiktok'
            )

        if data_dict.get("youtube") is not None:
            transformed["youtube"] = self.transform_social_media_data(
                data_dict["youtube"], 'youtube'
            )
        
        # 3. NUEVO: Combinar redes sociales
        social_merged = self.merge_social_media_data(
            transformed.get("instagram"),
            transformed.get("tiktok"),
            transformed.get("youtube")
        )
        if social_merged is not None:
            transformed["social_media_merged"] = social_merged
        
        # 4. NUEVO: Calcular agregados diarios
        if "sales" in transformed:
            daily_agg = self.calculate_daily_aggregates(
                transformed["sales"],
                social_merged
            )
            transformed["daily_aggregates"] = daily_agg
        
        # 5. Guardar todos los archivos procesados
        self._save_all_processed(transformed)

        logger.info("=" * 70)
        logger.info("TRANSFORMACIÓN COMPLETADA")
        logger.info("=" * 70)

        return transformed

    def transform_sales_data(self, df):
        """
        Transformación completa de ventas con limpieza profunda
        """
        logger.info("\n" + "="*70)
        logger.info("TRANSFORMANDO DATOS DE VENTAS")
        logger.info("="*70)
        
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # ========================================
        # PASO 1: GENERAR IDs FALTANTES
        # ========================================
        logger.info("\n[1/6] Generando identificadores...")
        
        if 'transaction_id' not in df_clean.columns:
            df_clean['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df_clean))]
            logger.info("transaction_id generado")
        
        if 'product_id' not in df_clean.columns:
            if 'product_name' in df_clean.columns:
                df_clean['product_id'] = pd.factorize(df_clean['product_name'])[0] + 1
                logger.info(f"product_id generado ({df_clean['product_id'].nunique()} productos)")
        
        if 'customer_id' not in df_clean.columns:
            if 'location' in df_clean.columns:
                df_clean = df_clean.sort_values('date').reset_index(drop=True)
                df_clean['_temp_group'] = (df_clean.index // 10).astype(str)
                df_clean['customer_id'] = pd.factorize(
                    df_clean['location'].astype(str) + '_' + df_clean['_temp_group']
                )[0] + 1
                df_clean = df_clean.drop('_temp_group', axis=1)
                logger.info(f"customer_id generado desde location ({df_clean['customer_id'].nunique()} clientes)")
            else:
                df_clean = df_clean.sort_values('date').reset_index(drop=True)
                df_clean['customer_id'] = (df_clean.index // 5) + 1
                logger.info(f"customer_id generado ({df_clean['customer_id'].nunique()} clientes)")
        
        # ========================================
        # PASO 2: CALCULAR MÉTRICAS DERIVADAS
        # ========================================
        logger.info("\n[2/6] Calculando métricas derivadas...")
        
        if 'amount' not in df_clean.columns:
            if 'quantity' in df_clean.columns and 'unit_price' in df_clean.columns:
                df_clean['amount'] = df_clean['quantity'] * df_clean['unit_price']
                if 'discount' in df_clean.columns:
                    df_clean['amount'] = df_clean['amount'] * (1 - df_clean['discount']/100)
                logger.info("amount calculado")
        
        if 'category' not in df_clean.columns and 'product_name' in df_clean.columns:
            df_clean['category'] = df_clean['product_name'].apply(self._infer_category)
            logger.info(f"category inferida ({df_clean['category'].nunique()} categorías)")
        
        # ========================================
        # PASO 3: LIMPIEZA DE DATOS
        # ========================================
        logger.info("\n[3/6] Limpiando datos...")
        
        # Duplicados
        duplicates = df_clean.duplicated(subset=['transaction_id']).sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=['transaction_id'])
            logger.info(f" Duplicados eliminados: {duplicates}")
        
        # Nulos críticos
        critical_cols = ['customer_id', 'date', 'product_name']
        nulls_before = df_clean[critical_cols].isnull().sum().sum()
        df_clean = df_clean.dropna(subset=critical_cols)
        if nulls_before > 0:
            logger.info(f"Registros con nulos eliminados: {nulls_before}")
        
        # Valores inválidos
        numeric_cols = ['quantity', 'unit_price', 'amount']
        for col in numeric_cols:
            if col in df_clean.columns:
                invalid = (df_clean[col] <= 0).sum()
                df_clean = df_clean[df_clean[col] > 0]
                if invalid > 0:
                    logger.info(f" Valores inválidos en '{col}': {invalid}")
        
        # ========================================
        # PASO 4: ELIMINAR OUTLIERS
        # ========================================
        logger.info("\n[4/6] Eliminando outliers...")
        
        if 'amount' in df_clean.columns:
            df_clean = self._remove_outliers(df_clean, 'amount')
        
        # ========================================
        # PASO 5: CREAR FEATURES TEMPORALES
        # ========================================
        logger.info("\n[5/6] Creando features temporales...")
        
        if 'date' in df_clean.columns:
            df_clean['year'] = df_clean['date'].dt.year
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['quarter'] = df_clean['date'].dt.quarter
            df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
            df_clean['day_name'] = df_clean['date'].dt.day_name()
            df_clean['week'] = df_clean['date'].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            logger.info("  Features temporales creadas")
        
        # ========================================
        # PASO 6: RESUMEN
        # ========================================
        logger.info("\n[6/6] Resumen de transformación...")
        
        final_count = len(df_clean)
        removed = initial_count - final_count
        removal_pct = (removed / initial_count * 100) if initial_count > 0 else 0
        
        logger.info(f"  Registros iniciales: {initial_count}")
        logger.info(f"  Registros finales: {final_count}")
        logger.info(f"  Registros eliminados: {removed} ({removal_pct:.1f}%)")
        logger.info(f"  Clientes únicos: {df_clean['customer_id'].nunique()}")
        logger.info(f"  Productos únicos: {df_clean['product_id'].nunique()}")
        
        logger.info("="*70)
        
        self.processed_data['sales'] = df_clean
        return df_clean
    
    def transform_social_media_data(self, df, platform):
        """
        VERSIÓN CORREGIDA: No recalcula engagement_rate si ya existe
        """
        logger.info(f"\nTransformando datos de {platform}...")

        df_clean = df.copy()

        # Eliminar duplicados
        id_cols = [col for col in ['post_id', 'video_id'] if col in df_clean.columns]
        if id_cols:
            initial = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=id_cols)
            if len(df_clean) < initial:
                logger.info(f"  Duplicados eliminados: {initial - len(df_clean)}")

        # Validar fechas
        if 'date' in df_clean.columns:
            nulls = df_clean['date'].isnull().sum()
            df_clean = df_clean.dropna(subset=['date'])
            if nulls > 0:
                logger.info(f"  Fechas nulas eliminadas: {nulls}")

        # ============================================
        # CRÍTICO: Solo calcular si NO existe
        # ============================================
        if 'engagement_rate' not in df_clean.columns:
            logger.info("  engagement_rate no existe, calculando...")

            if platform == 'instagram':
                if all(col in df_clean.columns for col in ['likes', 'comments', 'shares', 'followers']):
                    df_clean['engagement_rate'] = (
                        (df_clean['likes'] + df_clean['comments'] + df_clean['shares']) / 
                        df_clean['followers']
                    ).round(4)
                    logger.info("  engagement_rate calculado para Instagram")

            elif platform in ['tiktok', 'youtube']:
                if all(col in df_clean.columns for col in ['likes', 'comments', 'views']):
                    df_clean['engagement_rate'] = (
                        (df_clean['likes'] + df_clean['comments']) / 
                        df_clean['views']
                    ).round(4)

                    if 'shares' in df_clean.columns and platform == 'tiktok':
                        df_clean['engagement_rate'] = (
                            (df_clean['likes'] + df_clean['comments'] + df_clean['shares']) / 
                            df_clean['views']
                        ).round(4)

                    logger.info(f"  engagement_rate calculado para {platform}")
        else:
            logger.info(f"  engagement_rate ya existe (min: {df_clean['engagement_rate'].min():.4f}, max: {df_clean['engagement_rate'].max():.4f})")
            logger.info(f"  Variación: {df_clean['engagement_rate'].std():.4f}")

        # Features temporales
        if 'date' in df_clean.columns:
            df_clean['year'] = df_clean['date'].dt.year
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            df_clean['week'] = df_clean['date'].dt.isocalendar().week.astype(int) # Asegura que week sea int

        # Añadir plataforma
        df_clean['platform'] = platform

        logger.info(f"  {platform}: {len(df_clean)} registros procesados")

        return df_clean
    
    def merge_social_media_data(self, instagram_df, tiktok_df, youtube_df):
        """
        NUEVO: Combina datos de múltiples redes sociales
        """
        logger.info("\n" + "="*70)
        logger.info("COMBINANDO DATOS DE REDES SOCIALES")
        logger.info("="*70)
        
        dfs = []
        
        if instagram_df is not None and not instagram_df.empty:
            dfs.append(instagram_df)
            logger.info(f" Instagram: {len(instagram_df)} posts")
        
        if tiktok_df is not None and not tiktok_df.empty:
            dfs.append(tiktok_df)
            logger.info(f"TikTok: {len(tiktok_df)} videos")
        
        if youtube_df is not None and not youtube_df.empty:
            dfs.append(youtube_df)
            logger.info(f"YouTube: {len(youtube_df)} videos")
        
        if not dfs:
            logger.warning(" No hay datos de redes sociales para combinar")
            return None
        
        # Combinar
        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        
        logger.info(f"\n  Total combinado: {len(merged_df)} registros")
        logger.info(f"  Plataformas: {merged_df['platform'].unique().tolist()}")
        logger.info("="*70)
        
        self.processed_data['social_media_merged'] = merged_df
        return merged_df
    
    def calculate_daily_aggregates(self, sales_df, social_df):
        """
        VERSIÓN MEJORADA: Agregados diarios con métricas adicionales
        """
        logger.info("\n" + "="*70)
        logger.info("CALCULANDO AGREGADOS DIARIOS")
        logger.info("="*70)

        # Agregados de ventas por día
        logger.info("\nAgregando ventas por día...")
        sales_daily = sales_df.groupby('date').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'product_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()

        sales_daily.columns = [
            'date', 'total_sales', 'avg_order_value', 'num_transactions',
            'unique_customers', 'unique_products', 'total_units'
        ]

        logger.info(f"  Ventas agregadas: {len(sales_daily)} días")
        logger.info(f"  Rango ventas: ${sales_daily['total_sales'].min():.0f} - ${sales_daily['total_sales'].max():.0f}")

        # Agregados de redes sociales por día
        if social_df is not None and not social_df.empty:
            logger.info("\nAgregando redes sociales por día...")

            # Verificar variación de engagement ANTES de agregar
            logger.info(f"  Engagement original - std: {social_df['engagement_rate'].std():.4f}")

            social_daily = social_df.groupby('date').agg({
                'engagement_rate': ['mean', 'sum', 'std'],
                'likes': 'sum',
                'comments': 'sum',
                'platform': lambda x: ','.join(x.unique())
            }).reset_index()

            social_daily.columns = [
                'date', 'avg_engagement', 'total_engagement', 'std_engagement',
                'total_likes', 'total_comments', 'platforms'
            ]

            # Rellenar std nulos con 0
            social_daily['std_engagement'] = social_daily['std_engagement'].fillna(0)

            logger.info(f"  Redes sociales agregadas: {len(social_daily)} días")
            logger.info(f"  Engagement diario - std: {social_daily['avg_engagement'].std():.4f}")
            logger.info(f"  Rango engagement: {social_daily['avg_engagement'].min():.4f} - {social_daily['avg_engagement'].max():.4f}")

            # Combinar
            daily_merged = pd.merge(
                sales_daily,
                social_daily,
                on='date',
                how='left'
            )

            daily_merged = daily_merged.sort_values('date').reset_index(drop=True)

            # Rellenar nulos (mantener 0 para días sin datos)
            for col in ['avg_engagement', 'total_engagement', 'std_engagement',
                        'total_likes', 'total_comments']:
                if col in daily_merged.columns:
                    daily_merged[col] = daily_merged[col].fillna(0)
            
            daily_merged['total_sales'] = daily_merged['total_sales'].fillna(0)
            daily_merged['num_transactions'] = daily_merged['num_transactions'].fillna(0)

            # NUEVO: Crear métricas de interacción total
            daily_merged['total_interactions'] = (
                daily_merged['total_likes'] + daily_merged['total_comments']
            )

            logger.info(f"\n  Datos combinados: {len(daily_merged)} días")
            logger.info(f"  Días con ventas: {(daily_merged['total_sales'] > 0).sum()}")
            logger.info(f"  Días con engagement: {(daily_merged['avg_engagement'] > 0).sum()}")

        else:
            daily_merged = sales_daily
            logger.info("\n  Solo datos de ventas (sin redes sociales)")
            
        logger.info("="*70)
        self.processed_data['daily_aggregates'] = daily_merged
        return daily_merged
    
    def _remove_outliers(self, df, column, method='iqr', threshold=1.5):
        """Elimina outliers usando IQR"""
        
        if column not in df.columns:
            return df
        
        initial_count = len(df)
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_filtered = df[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]
        
        removed = initial_count - len(df_filtered)
        logger.info(f"Outliers eliminados: {removed} ({removed/initial_count*100:.1f}%)")
        
        return df_filtered
    
    def _infer_category(self, product_name):
        """Infiere categoría desde nombre del producto"""
        if pd.isna(product_name):
            return 'Other'
        
        product_lower = str(product_name).lower()
        
        if any(word in product_lower for word in ['protein', 'whey', 'casein', 'isolate']):
            return 'Protein'
        elif any(word in product_lower for word in ['creatine', 'bcaa', 'amino', 'glutamine', 'eaa']):
            return 'Amino Acids'
        elif any(word in product_lower for word in ['vitamin', 'mineral', 'multivitamin', 'omega', 'fish oil']):
            return 'Vitamins'
        elif any(word in product_lower for word in ['pre-workout', 'preworkout', 'pre workout', 'energy', 'caffeine']):
            return 'Performance'
        elif any(word in product_lower for word in ['protein bar', 'bar', 'snack', 'shake']):
            return 'Snacks'
        else:
            return 'Other'
    
    def _save_all_processed(self, transformed):
        """Guarda todos los datos procesados"""
        logger.info("\n" + "="*70)
        logger.info("GUARDANDO DATOS PROCESADOS")
        logger.info("="*70)
        
        output_path = Path('data/processed')
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, df in transformed.items():
            if df is not None and not df.empty:
                file_path = output_path / f'{name}_processed.csv'
                df.to_csv(file_path, index=False)
                logger.info(f"{file_path} ({len(df)} registros)")
        
        logger.info("="*70)