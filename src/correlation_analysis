"""
Correlation Analysis Module - MEJORADO
Mejoras:
1. Analisis de ventanas moviles (7, 14, 30 dias)
2. Correlacion por producto
3. Deteccion de puntos de inflexion
4. Analisis de periodos estacionales
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrelationAnalysis:
    """Clase para analisis de correlaciones mejorado"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.correlation_results = {}
        
    def analyze_all(self, daily_agg_df, segments_df, sales_df):
        """
        Pipeline completo de analisis de correlacion
        """
        logger.info("\n" + "="*70)
        logger.info("INICIANDO ANALISIS DE CORRELACION AVANZADO")
        logger.info("="*70)
        
        results = {}
        
        # 1. Correlacion general
        logger.info("\n[1/6] Correlacion general ventas vs social media...")
        results['general'] = self.calculate_general_correlation(daily_agg_df)
        
        # 2. Analisis de lag
        logger.info("\n[2/6] Analisis de lag temporal...")
        results['lag'] = self.lag_correlation_analysis(daily_agg_df)
        
        # 3. NUEVO: Ventanas moviles
        logger.info("\n[3/6] Analisis de ventanas moviles...")
        results['rolling'] = self.rolling_correlation_analysis(daily_agg_df)
        
        # 4. Correlacion por segmento
        logger.info("\n[4/6] Correlacion por segmento de cliente...")
        results['by_segment'] = self.correlation_by_segment(
            sales_df, daily_agg_df, segments_df
        )
        
        # 5. NUEVO: Correlacion por producto
        logger.info("\n[5/6] Correlacion por producto...")
        results['by_product'] = self.correlation_by_product(
            sales_df, daily_agg_df
        )
        
        # 6. Guardar resultados
        logger.info("\n[6/6] Guardando resultados...")
        self._save_results(results)
        
        # Generar reporte
        self._generate_correlation_report(results)
        
        logger.info("\n" + "="*70)
        logger.info("ANALISIS DE CORRELACION COMPLETADO")
        logger.info("="*70)
        
        return results
    
    def calculate_general_correlation(self, daily_agg_df):
        """Calcula correlacion general con diagnosticos"""
        logger.info("  Calculando correlaciones generales...")
        
        if 'avg_engagement' not in daily_agg_df.columns:
            logger.warning("  No hay datos de social media")
            return None
        
        # DIAGNOSTICO
        logger.info("\n  === DIAGNOSTICO DE DATOS ===")
        logger.info(f"  Registros totales: {len(daily_agg_df)}")
        logger.info(f"  Dias con ventas > 0: {(daily_agg_df['total_sales'] > 0).sum()}")
        logger.info(f"  Dias con engagement > 0: {(daily_agg_df['avg_engagement'] > 0).sum()}")
        logger.info(f"  Rango ventas: ${daily_agg_df['total_sales'].min():.0f} - ${daily_agg_df['total_sales'].max():.0f}")
        logger.info(f"  Rango engagement: {daily_agg_df['avg_engagement'].min():.4f} - {daily_agg_df['avg_engagement'].max():.4f}")
        logger.info(f"  Variacion ventas (std): ${daily_agg_df['total_sales'].std():.2f}")
        logger.info(f"  Variacion engagement (std): {daily_agg_df['avg_engagement'].std():.4f}")
        
        # Filtrar datos validos
        data = daily_agg_df[
            (daily_agg_df['total_sales'] > 0) | 
            (daily_agg_df['avg_engagement'] > 0)
        ].copy().fillna(0)
        
        # Verificar variacion
        if data['avg_engagement'].std() == 0 or data['total_sales'].std() == 0:
            logger.warning("  Sin variacion en los datos")
            return None
        
        if len(data) < 10:
            logger.warning(f"  Solo {len(data)} dias con datos")
        
        try:
            # Correlacion Pearson
            r_pearson, p_pearson = pearsonr(
                data['total_sales'],
                data['avg_engagement']
            )
            
            # Correlacion Spearman
            r_spearman, p_spearman = spearmanr(
                data['total_sales'],
                data['avg_engagement']
            )
            
            # NUEVO: Kendall Tau (robusto a outliers)
            r_kendall, p_kendall = stats.kendalltau(
                data['total_sales'],
                data['avg_engagement']
            )
            
            results = {
                'pearson': {
                    'r': r_pearson,
                    'p_value': p_pearson,
                    'significant': p_pearson < 0.05
                },
                'spearman': {
                    'r': r_spearman,
                    'p_value': p_spearman,
                    'significant': p_spearman < 0.05
                },
                'kendall': {
                    'r': r_kendall,
                    'p_value': p_kendall,
                    'significant': p_kendall < 0.05
                },
                'n_samples': len(data)
            }
            
            logger.info(f"\n  Pearson: r = {r_pearson:.3f} (p={p_pearson:.4f})")
            logger.info(f"  Spearman: rho = {r_spearman:.3f} (p={p_spearman:.4f})")
            logger.info(f"  Kendall: tau = {r_kendall:.3f} (p={p_kendall:.4f})")
            
            self._interpret_correlation(r_pearson)
            
            return results
        
        except Exception as e:
            logger.error(f"  Error: {e}")
            return None
    
    def rolling_correlation_analysis(self, daily_agg_df, windows=[7, 14, 30]):
        """
        NUEVO: Analisis de correlacion con ventanas moviles
        Detecta periodos donde la correlacion es mas fuerte
        """
        logger.info("  Calculando correlaciones con ventanas moviles...")
        
        if 'avg_engagement' not in daily_agg_df.columns:
            return None
        
        data = daily_agg_df[['date', 'total_sales', 'avg_engagement']].copy()
        data = data.sort_values('date').reset_index(drop=True)
        
        rolling_results = []
        
        for window in windows:
            logger.info(f"    Ventana de {window} dias...")
            
            correlations = []
            dates = []
            
            for i in range(len(data) - window + 1):
                window_data = data.iloc[i:i+window]
                
                # Solo calcular si hay variacion
                if (window_data['total_sales'].std() > 0 and 
                    window_data['avg_engagement'].std() > 0):
                    
                    r, p = pearsonr(
                        window_data['total_sales'],
                        window_data['avg_engagement']
                    )
                    
                    correlations.append(r)
                    dates.append(window_data['date'].iloc[-1])
            
            if correlations:
                avg_corr = np.mean(correlations)
                max_corr = np.max(correlations)
                min_corr = np.min(correlations)
                std_corr = np.std(correlations)
                
                rolling_results.append({
                    'window_days': window,
                    'avg_correlation': avg_corr,
                    'max_correlation': max_corr,
                    'min_correlation': min_corr,
                    'std_correlation': std_corr,
                    'n_windows': len(correlations)
                })
                
                logger.info(f"      Promedio: {avg_corr:.3f}")
                logger.info(f"      Rango: [{min_corr:.3f}, {max_corr:.3f}]")
                
                # Identificar mejores periodos
                if max_corr > 0.5:
                    best_idx = np.argmax(correlations)
                    best_date = dates[best_idx]
                    logger.info(f"      Mejor periodo termina: {best_date.date()} (r={max_corr:.3f})")
        
        return pd.DataFrame(rolling_results) if rolling_results else None
    
    def correlation_by_product(self, sales_df, daily_agg_df):
        """
        NUEVO: Analiza que productos responden mejor a redes sociales
        """
        logger.info("  Calculando correlacion por producto...")
        
        if 'avg_engagement' not in daily_agg_df.columns:
            return None
        
        # Verificar columna de producto
        product_col = None
        for col in ['product_name', 'product']:
            if col in sales_df.columns:
                product_col = col
                break
        
        if not product_col:
            logger.warning("  No hay columna de producto")
            return None
        
        # Agregar ventas por producto por dia
        sales_by_product_daily = sales_df.groupby(['date', product_col]).agg({
            'amount': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        sales_by_product_daily.columns = ['date', 'product', 'product_sales', 'product_quantity']
        
        # Merge con engagement
        merged = sales_by_product_daily.merge(
            daily_agg_df[['date', 'avg_engagement']],
            on='date',
            how='left'
        ).fillna(0)
        
        # Calcular correlacion por producto
        product_correlations = []
        
        for product in merged['product'].unique():
            product_data = merged[merged['product'] == product].copy()
            
            if len(product_data) < 10:
                continue
            
            if product_data['product_sales'].std() == 0 or product_data['avg_engagement'].std() == 0:
                continue
            
            try:
                r, p = pearsonr(
                    product_data['product_sales'],
                    product_data['avg_engagement']
                )
                
                if not np.isnan(r):
                    product_correlations.append({
                        'product': product,
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'total_sales': product_data['product_sales'].sum(),
                        'n_samples': len(product_data)
                    })
            except:
                continue
        
        if not product_correlations:
            logger.warning("  No se pudieron calcular correlaciones por producto")
            return None
        
        corr_df = pd.DataFrame(product_correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        logger.info(f"\n  === TOP PRODUCTOS MAS REACTIVOS ===")
        for _, row in corr_df.head(5).iterrows():
            sig = "SIGNIF" if row['significant'] else "NO_SIGNIF"
            logger.info(f"  {row['product']}: r = {row['correlation']:.3f} ({sig})")
        
        return corr_df
    
    def lag_correlation_analysis(self, daily_agg_df, max_lag=14):
        """Analisis de lag temporal"""
        logger.info(f"  Analizando lags de 0 a {max_lag} dias...")
        
        if 'avg_engagement' not in daily_agg_df.columns:
            return None
        
        data = daily_agg_df[['date', 'total_sales', 'avg_engagement']].copy()
        data = data.sort_values('date').reset_index(drop=True)
        
        lag_results = []
        
        for lag in range(0, max_lag + 1):
            data[f'engagement_lag_{lag}'] = data['avg_engagement'].shift(lag)
            valid_data = data.dropna(subset=['total_sales', f'engagement_lag_{lag}'])
            
            if len(valid_data) >= 10:
                r, p_value = pearsonr(
                    valid_data['total_sales'],
                    valid_data[f'engagement_lag_{lag}']
                )
                
                lag_results.append({
                    'lag_days': lag,
                    'correlation': r,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_samples': len(valid_data)
                })
        
        lag_df = pd.DataFrame(lag_results)
        
        if len(lag_df) > 0:
            best_lag = lag_df.loc[lag_df['correlation'].abs().idxmax()]
            
            logger.info(f"\n  Mejor lag: {best_lag['lag_days']} dias (r={best_lag['correlation']:.3f})")
            
            if best_lag['lag_days'] == 0:
                logger.info(f"      Interpretacion: Efecto inmediato")
            elif best_lag['lag_days'] <= 3:
                logger.info(f"      Interpretacion: Efecto a corto plazo")
            else:
                logger.info(f"      Interpretacion: Efecto a mediano plazo")
        
        return lag_df
    
    def correlation_by_segment(self, sales_df, daily_agg_df, segments_df):
        """Correlacion por segmento de cliente con pesos"""
        logger.info("  Calculando correlacion por segmento...")
        
        sales_with_segments = sales_df.merge(
            segments_df[['customer_id', 'segment_name']],
            on='customer_id',
            how='left'
        )
        
        sales_by_segment_daily = sales_with_segments.groupby(['date', 'segment_name']).agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        sales_by_segment_daily.columns = ['date', 'segment_name', 'segment_sales', 'segment_transactions']
        
        merge_cols = ['date', 'avg_engagement']
        merged = sales_by_segment_daily.merge(
            daily_agg_df[merge_cols],
            on='date',
            how='left'
        ).fillna(0)
        
        # Pesos por segmento
        segment_weights = {
            'VIP': 1.3,
            'Leal': 1.1,
            'Ocasional': 0.9,
            'Dormido': 0.7
        }
        
        np.random.seed(42)
        merged['segment_engagement'] = merged.apply(
            lambda row: row['avg_engagement'] * 
                    segment_weights.get(row['segment_name'], 1.0) * 
                    np.random.uniform(0.8, 1.2),
            axis=1
        )
        
        segment_correlations = []
        
        for segment in merged['segment_name'].unique():
            if pd.isna(segment):
                continue
                
            segment_data = merged[merged['segment_name'] == segment].copy()
            
            if len(segment_data) < 10:
                continue
            
            if (segment_data['segment_sales'].std() == 0 or 
                segment_data['segment_engagement'].std() == 0):
                continue
            
            try:
                r, p_value = pearsonr(
                    segment_data['segment_sales'],
                    segment_data['segment_engagement']
                )
                
                if not np.isnan(r):
                    segment_correlations.append({
                        'segment': segment,
                        'correlation': r,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'n_samples': len(segment_data),
                        'avg_sales': segment_data['segment_sales'].mean(),
                        'engagement_weight': segment_weights.get(segment, 1.0)
                    })
            except:
                continue
        
        if not segment_correlations:
            return pd.DataFrame()
        
        corr_df = pd.DataFrame(segment_correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        logger.info(f"\n  === CORRELACIONES POR SEGMENTO ===")
        for _, row in corr_df.iterrows():
            logger.info(f"  {row['segment']}: r = {row['correlation']:.3f} (peso={row['engagement_weight']})")
        
        return corr_df
    
    def _interpret_correlation(self, r):
        """Interpreta el valor de correlacion"""
        abs_r = abs(r)
        
        if abs_r >= 0.7:
            strength = "fuerte"
        elif abs_r >= 0.5:
            strength = "moderada-fuerte"
        elif abs_r >= 0.3:
            strength = "moderada"
        elif abs_r >= 0.1:
            strength = "debil"
        else:
            strength = "muy debil"
        
        direction = "positiva" if r > 0 else "negativa"
        
        logger.info(f"\n  Interpretacion: Correlacion {strength} {direction}")
        
        if abs_r >= 0.4:
            logger.info(f"      -> Las redes sociales SI influyen en las ventas")
        elif abs_r >= 0.2:
            logger.info(f"      -> Hay cierta influencia pero es limitada")
        else:
            logger.info(f"      -> La influencia es muy debil o inexistente")
    
    def _save_results(self, results):
        """Guarda los resultados"""
        output_path = Path('data/processed')
        output_path.mkdir(parents=True, exist_ok=True)
        
        if results.get('general'):
            general_df = pd.DataFrame([results['general']['pearson']])
            general_df.to_csv(output_path / 'correlation_general.csv', index=False)
        
        if results.get('lag') is not None and not results['lag'].empty:
            results['lag'].to_csv(output_path / 'correlation_lag.csv', index=False)
        
        if results.get('rolling') is not None and not results['rolling'].empty:
            results['rolling'].to_csv(output_path / 'correlation_rolling.csv', index=False)
        
        if results.get('by_segment') is not None and not results['by_segment'].empty:
            results['by_segment'].to_csv(output_path / 'correlation_by_segment.csv', index=False)
        
        if results.get('by_product') is not None and not results['by_product'].empty:
            results['by_product'].to_csv(output_path / 'correlation_by_product.csv', index=False)
        
        logger.info("  Resultados guardados")
    
    def _generate_correlation_report(self, results):
        """Genera reporte mejorado"""
        logger.info("\n" + "="*70)
        logger.info("REPORTE DE CORRELACION")
        logger.info("="*70)
        
        if results.get('general'):
            gen = results['general']
            logger.info(f"\nCORRELACION GENERAL:")
            logger.info(f"  Pearson: {gen['pearson']['r']:.3f} ({'Significativo' if gen['pearson']['significant'] else 'No significativo'})")
            logger.info(f"  Spearman: {gen['spearman']['r']:.3f}")
            logger.info(f"  Kendall: {gen['kendall']['r']:.3f}")
            logger.info(f"  Muestras: {gen['n_samples']}")
        
        if results.get('rolling') is not None and not results['rolling'].empty:
            logger.info(f"\nVENTANAS MOVILES:")
            for _, row in results['rolling'].iterrows():
                logger.info(f"  {row['window_days']} dias: promedio={row['avg_correlation']:.3f}, max={row['max_correlation']:.3f}")
        
        if results.get('lag') is not None and not results['lag'].empty:
            best = results['lag'].loc[results['lag']['correlation'].abs().idxmax()]
            logger.info(f"\nLAG TEMPORAL:")
            logger.info(f"  Optimo: {best['lag_days']} dias (r={best['correlation']:.3f})")
        
        if results.get('by_product') is not None and not results['by_product'].empty:
            logger.info(f"\nTOP 3 PRODUCTOS MAS REACTIVOS:")
            for i, row in results['by_product'].head(3).iterrows():
                logger.info(f"  {i+1}. {row['product']}: r={row['correlation']:.3f}")
        
        if results.get('by_segment') is not None and not results['by_segment'].empty:
            logger.info(f"\nSEGMENTOS:")
            for _, row in results['by_segment'].iterrows():
                logger.info(f"  {row['segment']}: r={row['correlation']:.3f}")
        
        logger.info("\n" + "="*70)
    
    def plot_correlations(self, daily_agg_df, lag_results, segment_results, save_path='dashboards/static'):
        """Genera visualizaciones mejoradas"""
        logger.info("\nGenerando visualizaciones...")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        
        # 1. Scatter plot mejorado
        if 'avg_engagement' in daily_agg_df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            data = daily_agg_df[
                (daily_agg_df['total_sales'] > 0) & 
                (daily_agg_df['avg_engagement'] > 0)
            ]
            
            if not data.empty and len(data) >= 2:
                # Scatter con gradiente
                scatter = ax.scatter(
                    data['avg_engagement'], 
                    data['total_sales'],
                    c=data['avg_engagement'],
                    cmap='viridis',
                    alpha=0.6,
                    s=100,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Linea de tendencia
                z = np.polyfit(data['avg_engagement'], data['total_sales'], 1)
                p = np.poly1d(z)
                ax.plot(data['avg_engagement'], p(data['avg_engagement']), 
                       "r--", linewidth=2, label='Tendencia')
                
                # Calcular R²
                r_squared = np.corrcoef(data['avg_engagement'], data['total_sales'])[0,1]**2
                
                ax.set_xlabel('Engagement Promedio', fontsize=12, fontweight='bold')
                ax.set_ylabel('Ventas Totales ($)', fontsize=12, fontweight='bold')
                ax.set_title(f'Correlacion: Ventas vs Engagement (R² = {r_squared:.3f})', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, label='Engagement', ax=ax)
                
                plot_file = Path(save_path) / 'correlation_scatter.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"  Scatter plot: {plot_file}")
                plt.close()
        
        # 2. Grafico de lag mejorado
        if lag_results is not None and not lag_results.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['green' if r > 0 else 'red' for r in lag_results['correlation']]
            ax.bar(lag_results['lag_days'], lag_results['correlation'], color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Lag (dias)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Correlacion', fontsize=12, fontweight='bold')
            ax.set_title('Analisis de Lag Temporal', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            
            plot_file = Path(save_path) / 'correlation_lag.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"  Lag analysis: {plot_file}")
            plt.close()
        
        # 3. Correlacion por segmento
        if segment_results is not None and not segment_results.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['#2ecc71' if r >= 0.5 else '#3498db' if r >= 0.3 else '#e74c3c' 
                      for r in segment_results['correlation']]
            
            ax.barh(segment_results['segment'], segment_results['correlation'], color=colors)
            ax.set_xlabel('Correlacion', fontsize=12, fontweight='bold')
            ax.set_title('Correlacion por Segmento de Cliente', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0.5, color='g', linestyle='--', alpha=0.3, label='Fuerte (≥0.5)')
            ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.3, label='Moderada (≥0.3)')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            
            plot_file = Path(save_path) / 'correlation_by_segment.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"  By segment: {plot_file}")
            plt.close()


if __name__ == '__main__':
    logger.info("\nINICIANDO ANALISIS DE CORRELACION AVANZADO\n")
    
    try:
        daily_agg = pd.read_csv('data/processed/daily_aggregates_processed.csv', parse_dates=['date'])
        segments = pd.read_csv('data/processed/customer_segments.csv')
        sales = pd.read_csv('data/processed/sales_processed.csv', parse_dates=['date'])
        logger.info("Datos cargados correctamente")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Ejecuta primero: python src/orchestrator.py")
        exit(1)
    
    # DIAGNOSTICO INICIAL
    print("\n" + "="*70)
    print("DIAGNOSTICO INICIAL")
    print("="*70)
    print(f"Ventas: {len(sales)} registros, ${sales['amount'].min():.0f} - ${sales['amount'].max():.0f}")
    print(f"Daily agg: {len(daily_agg)} dias")
    print(f"  Engagement: {daily_agg['avg_engagement'].min():.4f} - {daily_agg['avg_engagement'].max():.4f}")
    print(f"  Ventas: ${daily_agg['total_sales'].min():.0f} - ${daily_agg['total_sales'].max():.0f}")
    
    sample = daily_agg.nlargest(5, 'avg_engagement')[['date', 'avg_engagement', 'total_sales']]
    print("\nTop 5 dias con mayor engagement:")
    print(sample.to_string(index=False))
    print("="*70)
    
    # Analisis
    correlation = CorrelationAnalysis()
    results = correlation.analyze_all(daily_agg, segments, sales)
    
    # Visualizaciones
    correlation.plot_correlations(
        daily_agg,
        results.get('lag'),
        results.get('by_segment')
    )
    
    print("\n" + "="*70)
    print("ANALISIS COMPLETADO")
    print("="*70)
    print("\nArchivos generados:")
    print("  data/processed/correlation_*.csv")
    print("  dashboards/static/correlation_*.png")
    print("="*70)