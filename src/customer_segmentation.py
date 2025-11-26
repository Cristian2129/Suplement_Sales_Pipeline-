"""
Customer Segmentation Module
Análisis RFM, K-means clustering y PCA
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """Clase para segmentación de clientes"""
    
    def __init__(self):
        """Inicializa el segmentador"""
        self.rfm_data = None
        self.segments = None
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        
    def segment_customers(self, sales_df, n_clusters=4):
        """
        Pipeline completo de segmentación
        
        Args:
            sales_df: DataFrame con datos de ventas
            n_clusters: Número de segmentos (default: 4)
            
        Returns:
            dict con rfm, segments, pca_coords
        """
        logger.info("\n" + "="*70)
        logger.info("INICIANDO SEGMENTACIÓN DE CLIENTES")
        logger.info("="*70)
        
        # 1. Calcular RFM
        logger.info("\n[1/5] Calculando métricas RFM...")
        rfm = self.calculate_rfm(sales_df)
        
        # 2. K-means Clustering
        logger.info("\n[2/5] Aplicando K-means clustering...")
        segments = self.perform_clustering(rfm, n_clusters)
        
        # 3. Etiquetar segmentos
        logger.info("\n[3/5] Etiquetando segmentos...")
        segments = self.label_segments(segments)
        
        # 4. PCA para visualización
        logger.info("\n[4/5] Aplicando PCA para visualización...")
        pca_coords = self.apply_pca(rfm)
        
        # 5. Guardar resultados
        logger.info("\n[5/5] Guardando resultados...")
        self._save_results(rfm, segments)
        
        # Generar reporte
        self._generate_segment_report(segments)
        
        logger.info("\n" + "="*70)
        logger.info("SEGMENTACIÓN COMPLETADA")
        logger.info("="*70)
        
        return {
            'rfm': rfm,
            'segments': segments,
            'pca_coords': pca_coords
        }
        
    def calculate_rfm(self, sales_df):
        """
        Calcula métricas RFM por cliente
        
        RFM:
        - Recency: Días desde última compra
        - Frequency: Número de transacciones
        - Monetary: Total gastado
        """
        logger.info("  Calculando métricas RFM por cliente...")
        
        # Fecha de referencia (la más reciente en los datos)
        reference_date = sales_df['date'].max()
        logger.info(f"  Fecha de referencia: {reference_date.date()}")
        
        # Calcular RFM
        rfm = sales_df.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',                         # Frequency
            'amount': 'sum'                                    # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Métricas adicionales
        rfm['avg_order_value'] = sales_df.groupby('customer_id')['amount'].mean().values
        rfm['total_quantity'] = sales_df.groupby('customer_id')['quantity'].sum().values
        
        # Calcular product_diversity (cuántos productos diferentes compró)
        product_diversity = sales_df.groupby('customer_id')['product_id'].nunique()
        rfm['product_diversity'] = product_diversity.values
        
        logger.info(f"  RFM calculado para {len(rfm)} clientes")
        logger.info(f"  Recency: {rfm['recency'].min():.0f} - {rfm['recency'].max():.0f} días")
        logger.info(f"  Frequency: {rfm['frequency'].min():.0f} - {rfm['frequency'].max():.0f} compras")
        logger.info(f"  Monetary: ${rfm['monetary'].min():.2f} - ${rfm['monetary'].max():.2f}")
        
        self.rfm_data = rfm
        return rfm
    
    def perform_clustering(self, rfm_df, n_clusters=4):
        """
        Aplica K-means clustering
        
        Args:
            rfm_df: DataFrame con métricas RFM
            n_clusters: Número de clusters
            
        Returns:
            DataFrame con columna 'segment' añadida
        """
        logger.info(f"  Aplicando K-means con {n_clusters} clusters...")
        
        # Seleccionar features para clustering
        features = ['recency', 'frequency', 'monetary']
        X = rfm_df[features].values
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm_df['segment'] = self.kmeans.fit_predict(X_scaled)
        
        # Calcular métricas del clustering
        inertia = self.kmeans.inertia_
        logger.info(f"  Clustering completado (inertia: {inertia:.2f})")
        
        # Distribución de segmentos
        segment_dist = rfm_df['segment'].value_counts().sort_index()
        logger.info(f"  Distribución de segmentos:")
        for seg, count in segment_dist.items():
            pct = (count / len(rfm_df)) * 100
            logger.info(f"    Segmento {seg}: {count} clientes ({pct:.1f}%)")
        
        self.segments = rfm_df
        return rfm_df
    
    def label_segments(self, segments_df):
        """
        Etiqueta los segmentos con nombres descriptivos
        basándose en sus características RFM
        """
        logger.info("  Analizando características de cada segmento...")
        
        # Calcular promedios por segmento
        segment_profiles = segments_df.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        logger.info("\n  Perfiles de segmentos:")
        logger.info(segment_profiles.to_string())
        
        # Etiquetar segmentos basándose en características
        segment_labels = {}
        
        for seg in segments_df['segment'].unique():
            seg_data = segment_profiles.loc[seg]
            
            # Lógica de etiquetado
            if seg_data['frequency'] >= segments_df['frequency'].quantile(0.75) and \
               seg_data['monetary'] >= segments_df['monetary'].quantile(0.75):
                segment_labels[seg] = 'VIP'
            elif seg_data['frequency'] >= segments_df['frequency'].median() and \
                 seg_data['recency'] <= segments_df['recency'].median():
                segment_labels[seg] = 'Leal'
            elif seg_data['recency'] >= segments_df['recency'].quantile(0.75):
                segment_labels[seg] = 'Dormido'
            else:
                segment_labels[seg] = 'Ocasional'
        
        # Aplicar etiquetas
        segments_df['segment_name'] = segments_df['segment'].map(segment_labels)
        
        logger.info("\n  Etiquetas asignadas:")
        for seg, label in segment_labels.items():
            count = (segments_df['segment'] == seg).sum()
            logger.info(f"    Segmento {seg} -> '{label}' ({count} clientes)")
        
        return segments_df
    
    def apply_pca(self, rfm_df, n_components=2):
        """
        Aplica PCA para reducción dimensional (visualización)
        
        Args:
            rfm_df: DataFrame con métricas RFM
            n_components: Número de componentes (2 o 3)
            
        Returns:
            array con coordenadas PCA
        """
        logger.info(f"  Aplicando PCA ({n_components} componentes)...")
        
        # Features para PCA
        features = ['recency', 'frequency', 'monetary']
        X = rfm_df[features].values
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        self.pca = PCA(n_components=n_components)
        pca_coords = self.pca.fit_transform(X_scaled)
        
        # Varianza explicada
        variance_explained = self.pca.explained_variance_ratio_
        total_variance = variance_explained.sum()
        
        logger.info(f"  PCA completado")
        logger.info(f"  Varianza explicada por componente:")
        for i, var in enumerate(variance_explained):
            logger.info(f"    PC{i+1}: {var*100:.2f}%")
        logger.info(f"  Varianza total explicada: {total_variance*100:.2f}%")
        
        return pca_coords
    
    def _save_results(self, rfm_df, segments_df):
        """Guarda los resultados del análisis"""
        output_path = Path('data/processed')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar RFM
        rfm_file = output_path / 'rfm_analysis.csv'
        rfm_df.to_csv(rfm_file, index=False)
        logger.info(f"  RFM guardado: {rfm_file}")
        
        # Guardar segmentos
        segments_file = output_path / 'customer_segments.csv'
        segments_df.to_csv(segments_file, index=False)
        logger.info(f"  Segmentos guardados: {segments_file}")
    
    def _generate_segment_report(self, segments_df):
        """Genera reporte detallado de segmentos"""
        logger.info("\n" + "="*70)
        logger.info("REPORTE DE SEGMENTACIÓN")
        logger.info("="*70)
        
        for segment_name in segments_df['segment_name'].unique():
            seg_data = segments_df[segments_df['segment_name'] == segment_name]
            
            logger.info(f"\nSEGMENTO: {segment_name}")
            logger.info(f"  Tamaño: {len(seg_data)} clientes ({len(seg_data)/len(segments_df)*100:.1f}%)")
            logger.info(f"\n  Características promedio:")
            logger.info(f"    Recency: {seg_data['recency'].mean():.1f} días")
            logger.info(f"    Frequency: {seg_data['frequency'].mean():.1f} compras")
            logger.info(f"    Monetary: ${seg_data['monetary'].mean():.2f}")
            logger.info(f"    AOV: ${seg_data['avg_order_value'].mean():.2f}")
            logger.info(f"    Product Diversity: {seg_data['product_diversity'].mean():.1f} productos")
            
            # Valor total del segmento
            total_value = seg_data['monetary'].sum()
            logger.info(f"\n  Valor total del segmento: ${total_value:,.2f}")
            
            # Recomendaciones
            logger.info(f"\n  Estrategia recomendada:")
            if segment_name == 'VIP':
                logger.info("    -> Programas de fidelidad premium")
                logger.info("    -> Lanzamientos exclusivos")
                logger.info("    -> Atención personalizada")
            elif segment_name == 'Leal':
                logger.info("    -> Incentivos para aumentar ticket promedio")
                logger.info("    -> Cross-selling de productos complementarios")
                logger.info("    -> Programas de referidos")
            elif segment_name == 'Ocasional':
                logger.info("    -> Campañas de retención")
                logger.info("    -> Descuentos en segunda compra")
                logger.info("    -> Email marketing educativo")
            elif segment_name == 'Dormido':
                logger.info("    -> Campañas de reactivación")
                logger.info("    -> Descuentos especiales de 'regreso'")
                logger.info("    -> Encuestas para entender por qué se fueron")
        
        logger.info("\n" + "="*70)
    
    def plot_segments(self, segments_df, pca_coords, save_path='dashboards/static'):
        """
        Genera visualizaciones de los segmentos
        
        Args:
            segments_df: DataFrame con segmentos
            pca_coords: Coordenadas PCA
            save_path: Ruta para guardar gráficos
        """
        logger.info("\nGenerando visualizaciones...")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. Scatter plot PCA
        plt.figure(figsize=(12, 8))
        
        for segment_name in segments_df['segment_name'].unique():
            mask = segments_df['segment_name'] == segment_name
            plt.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                label=segment_name,
                alpha=0.6,
                s=100
            )
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Segmentación de Clientes (PCA)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_file = Path(save_path) / 'segments_pca.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Gráfico PCA guardado: {plot_file}")
        plt.close()
        
        # 2. Distribución de segmentos
        plt.figure(figsize=(10, 6))
        segment_counts = segments_df['segment_name'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        segment_counts.plot(kind='bar', color=colors)
        plt.title('Distribución de Segmentos de Clientes', fontsize=14, fontweight='bold')
        plt.xlabel('Segmento')
        plt.ylabel('Número de Clientes')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plot_file = Path(save_path) / 'segments_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Gráfico de distribución guardado: {plot_file}")
        plt.close()
        
        # 3. Características por segmento
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        segment_profiles = segments_df.groupby('segment_name').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        })
        
        segment_profiles['recency'].plot(kind='bar', ax=axes[0], color='#FF6B6B')
        axes[0].set_title('Recency Promedio')
        axes[0].set_ylabel('Días')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        segment_profiles['frequency'].plot(kind='bar', ax=axes[1], color='#4ECDC4')
        axes[1].set_title('Frequency Promedio')
        axes[1].set_ylabel('Número de Compras')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
        segment_profiles['monetary'].plot(kind='bar', ax=axes[2], color='#45B7D1')
        axes[2].set_title('Monetary Promedio')
        axes[2].set_ylabel('Valor Total ($)')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_file = Path(save_path) / 'segments_rfm_profiles.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Gráfico de perfiles RFM guardado: {plot_file}")
        plt.close()


if __name__ == '__main__':
    logger.info("\n INICIANDO ANÁLISIS DE SEGMENTACIÓN DE CLIENTES\n")
    
    # Cargar datos
    try:
        sales_df = pd.read_csv('data/processed/sales_processed.csv', parse_dates=['date'])
        logger.info(f"Datos cargados: {len(sales_df)} transacciones")
    except FileNotFoundError:
        logger.error("Archivo sales_processed.csv no encontrado")
        logger.error("  Ejecuta primero: python src/orchestrator.py")
        exit(1)
    
    # Segmentación
    segmentation = CustomerSegmentation()
    result = segmentation.segment_customers(sales_df, n_clusters=4)
    
    # Generar visualizaciones
    segmentation.plot_segments(
        result['segments'],
        result['pca_coords']
    )
    
    print("\n" + "="*70)
    print("SEGMENTACIÓN COMPLETADA")
    print("="*70)
    print("\nArchivos generados:")
    print("  data/processed/rfm_analysis.csv")
    print("  data/processed/customer_segments.csv")
    print("  dashboards/static/segments_*.png")
    print("="*70)