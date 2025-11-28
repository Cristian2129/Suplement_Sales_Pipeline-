import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import psutil
import time
import subprocess
import threading
from PIL import Image
import json
import sys
import os

st.set_page_config(
    page_title="Pipeline de Ventas de Suplementos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def get_system_metrics():
    """Obtiene metricas del sistema"""
    try:
        if os.name == 'nt':
            root_path = 'C:\\'
        else:
            root_path = '/'
        
        sys_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'cpu_count': psutil.cpu_count(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_percent': psutil.disk_usage(root_path).percent,
            'disk_used_gb': psutil.disk_usage(root_path).used / (1024**3),
            'disk_total_gb': psutil.disk_usage(root_path).total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
        return sys_metrics
    except Exception as e:
        st.error(f"Error obteniendo metricas: {e}")
        return {
            'cpu_percent': 0, 'cpu_count': 0,
            'memory_percent': 0, 'memory_used_gb': 0, 'memory_total_gb': 0,
            'disk_percent': 0, 'disk_used_gb': 0, 'disk_total_gb': 0,
            'timestamp': datetime.now().isoformat()
        }


@st.cache_data(ttl=30)
def load_data(filepath):
    """Carga datos con cache de 30 segundos"""
    try:
        if Path(filepath).exists():
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                return pd.read_json(filepath)
    except Exception as e:
        st.error(f"Error cargando {filepath}: {e}")
    return None


def save_execution_history(status, duration, error=None):
    """Guarda historial de ejecuciones"""
    history_file = Path('data/execution_history.json')
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    execution = {
        'timestamp': datetime.now().isoformat(),
        'status': status,
        'duration_seconds': duration,
        'error': error
    }
    
    # Cargar historial existente
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Agregar nueva ejecucion
    history.append(execution)
    
    # Mantener solo ultimas 50 ejecuciones
    history = history[-50:]
    
    # Guardar
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def get_execution_history():
    """Lee historial de ejecuciones"""
    history_file = Path('data/execution_history.json')
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


# ==========================================
# EJECUTOR DE PIPELINE - CORREGIDO
# ==========================================

def run_pipeline():
    """
    Ejecuta el pipeline completo usando subprocess
    """
    try:
        start_time = time.time()
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Inicializando pipeline...')
        progress_bar.progress(10)
        
        # Obtener ruta al orchestrator
        src_path = Path(__file__).parent.parent / 'src'
        orchestrator_path = src_path / 'orchestrator.py'
        
        if not orchestrator_path.exists():
            st.error(f"No se encontro orchestrator.py en {orchestrator_path}")
            return False
        
        status_text.text('Ejecutando pipeline...')
        progress_bar.progress(30)
        
        # Ejecutar como subprocess
        result = subprocess.run(
            [sys.executable, str(orchestrator_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        progress_bar.progress(90)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            save_execution_history('success', duration)
            progress_bar.progress(100)
            status_text.text('Completado!')
            st.success(f'Pipeline completado en {duration:.1f}s')
            
            # Mostrar ultimas lineas del output
            with st.expander("Ver log de ejecucion"):
                output_lines = result.stdout.split('\n')
                st.code('\n'.join(output_lines[-30:]))
            
            return True
        else:
            save_execution_history('failed', duration, result.stderr)
            st.error(f'Pipeline fallo (codigo: {result.returncode})')
            
            with st.expander("Ver error"):
                st.code(result.stderr)
            
            return False
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        save_execution_history('timeout', duration, 'Timeout after 5 minutes')
        st.error('Pipeline timeout (>5 minutos)')
        return False
        
    except Exception as e:
        duration = time.time() - start_time
        save_execution_history('error', duration, str(e))
        st.error(f'Error: {e}')
        return False
    finally:
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass


# ==========================================
# DASHBOARD PRINCIPAL
# ==========================================

def main():
    # Header mejorado
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.title("Observatorio del Pipeline")
        st.caption("Observabilidad completa del pipeline de datos")
    with col2:
        if st.button("Actualizar", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        if st.button("Ejecutar Pipeline", use_container_width=True, type="primary"):
            run_pipeline()
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # ==========================================
    # PESTAÑAS PRINCIPALES
    # ==========================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen", 
        "Rendimiento",
        "Resultados del Analisis",
        "Registros",
        "Historial"
    ])
    
    # ==========================================
    # PESTAÑA 1: RESUMEN
    # ==========================================
    with tab1:
        st.header("Resumen del Estado del Pipeline")
        
        # Estado de archivos
        files = {
            'ventas': 'data/processed/sales_processed.csv',
            'redes_sociales': 'data/processed/social_media_merged_processed.csv',
            'agregados_diarios': 'data/processed/daily_aggregates_processed.csv',
            'segmentos': 'data/processed/customer_segments.csv',
            'correlacion': 'data/processed/correlation_general.csv'
        }
        
        total = len(files)
        exists = sum(1 for f in files.values() if Path(f).exists())
        
        # Metricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status = "OK" if exists == total else "Parcial"
            st.metric("Estado del Pipeline", status, f"{exists}/{total} datasets")
        
        with col2:
            sales = load_data(files['ventas'])
            records = len(sales) if sales is not None else 0
            st.metric("Registros Totales", f"{records:,}")
        
        with col3:
            daily = load_data(files['agregados_diarios'])
            days = len(daily) if daily is not None else 0
            st.metric("Dias Analizados", f"{days:,}")
        
        with col4:
            corr = load_data(files['correlacion'])
            if corr is not None and len(corr) > 0:
                r = corr['r'].iloc[0]
                st.metric("Correlacion", f"{r:.3f}")
            else:
                st.metric("Correlacion", "N/A")
        
        with col5:
            # Ultima ejecucion
            history = get_execution_history()
            if history:
                last = history[-1]
                last_time = datetime.fromisoformat(last['timestamp'])
                time_ago = (datetime.now() - last_time).seconds // 60
                status_icon = "✅" if last['status'] == 'success' else "❌"
                st.metric("Ultima Ejecucion", f"{time_ago}m atras", status_icon)
            else:
                st.metric("Ultima Ejecucion", "Nunca", "➖")
        
        st.markdown("---")
        
        # Estadisticas rapidas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Resumen de Datos")
            
            if sales is not None:
                st.write(f"**Ventas**: {len(sales):,} transacciones")
                if 'amount' in sales.columns:
                    total_revenue = sales['amount'].sum()
                    st.write(f"**Ingresos**: ${total_revenue:,.0f}")
                if 'customer_id' in sales.columns:
                    st.write(f"**Clientes**: {sales['customer_id'].nunique():,}")
                if 'product_name' in sales.columns:
                    st.write(f"**Productos**: {sales['product_name'].nunique()}")
            
            social = load_data(files['redes_sociales'])
            if social is not None:
                st.write(f"**Publicaciones Sociales**: {len(social):,}")
                if 'platform' in social.columns:
                    platforms = ', '.join(social['platform'].unique())
                    st.write(f"**Plataformas**: {platforms}")
        
        with col2:
            st.subheader("Informacion Clave")
            
            if corr is not None and len(corr) > 0:
                r = corr['r'].iloc[0]
                sig = "Significativa" if corr['significant'].iloc[0] else "No significativa"
                direction = "Positiva" if r > 0 else "Negativa"
                st.write(f"**Correlacion**: {r:.3f} ({direction})")
                st.write(f"**Significancia**: {sig}")
            
            segments = load_data(files['segmentos'])
            if segments is not None and 'segment_name' in segments.columns:
                top_segment = segments['segment_name'].value_counts().index[0]
                count = segments['segment_name'].value_counts().iloc[0]
                st.write(f"**Segmento Mas Grande**: {top_segment} ({count})")
            
            lag = load_data('data/processed/correlation_lag.csv')
            if lag is not None and len(lag) > 0:
                best = lag.loc[lag['correlation'].abs().idxmax()]
                st.write(f"**Lag Optimo**: {best['lag_days']:.0f} dias")
        
        with col3:
            st.subheader("Estado de Archivos")
            
            for name, path in files.items():
                if Path(path).exists():
                    size = Path(path).stat().st_size / 1024
                    modified = datetime.fromtimestamp(Path(path).stat().st_mtime)
                    time_ago = datetime.now() - modified
                    
                    if time_ago.seconds < 3600:
                        time_str = f"{time_ago.seconds // 60}m atras"
                    else:
                        time_str = f"{time_ago.seconds // 3600}h atras"
                    
                    st.write(f"**{name}**: {size:.0f}KB ({time_str})")
                else:
                    st.write(f"**{name}**: Faltante")
    
    # ==========================================
    # PESTAÑA 2: RENDIMIENTO
    # ==========================================
    with tab2:
        st.header("Metricas de Rendimiento")
        
        history = get_execution_history()
        
        if history:
            # Convertir a DataFrame
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            df_history = df_history.sort_values('timestamp')
            
            # Grafico de tiempos de ejecucion
            st.subheader("Tiempos de Ejecucion")
            
            fig = px.line(df_history, x='timestamp', y='duration_seconds',
                         title='Duracion de Ejecucion del Pipeline en el Tiempo',
                         markers=True,
                         color='status',
                         color_discrete_map={'success': 'green', 'failed': 'red', 'error': 'orange'})
            fig.update_layout(xaxis_title='Fecha', yaxis_title='Duracion (segundos)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadisticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_duration = df_history['duration_seconds'].mean()
                st.metric("Duracion Promedio", f"{avg_duration:.1f}s")
            
            with col2:
                min_duration = df_history['duration_seconds'].min()
                st.metric("Mas Rapida", f"{min_duration:.1f}s")
            
            with col3:
                max_duration = df_history['duration_seconds'].max()
                st.metric("Mas Lenta", f"{max_duration:.1f}s")
            
            with col4:
                success_rate = (df_history['status'] == 'success').sum() / len(df_history) * 100
                st.metric("Tasa de Exito", f"{success_rate:.0f}%")
        else:
            st.info("No hay historial de ejecuciones disponible. Ejecute el pipeline primero!")
        
        st.markdown("---")
        
        # Tamanos de archivos
        st.subheader("Tamanos de Archivos")
        
        files = {
            'ventas': 'data/processed/sales_processed.csv',
            'redes_sociales': 'data/processed/social_media_merged_processed.csv',
            'agregados_diarios': 'data/processed/daily_aggregates_processed.csv',
            'segmentos': 'data/processed/customer_segments.csv'
        }
        
        file_data = []
        for name, path in files.items():
            if Path(path).exists():
                size_kb = Path(path).stat().st_size / 1024
                file_data.append({'Dataset': name, 'Tamaño (KB)': size_kb})
        
        if file_data:
            df_sizes = pd.DataFrame(file_data)
            fig = px.bar(df_sizes, x='Dataset', y='Tamaño (KB)',
                        color='Tamaño (KB)',
                        color_continuous_scale='Blues',
                        title='Tamanos de Datasets')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # PESTAÑA 3: RESULTADOS DEL ANALISIS
    # ==========================================
    with tab3:
        st.header("Resultados del Analisis")
        
        st.subheader("Visualizaciones")
        
        images = {
            'Dispersion de Correlacion': 'dashboards/static/correlation_scatter.png',
            'Correlacion por Segmento': 'dashboards/static/correlation_by_segment.png',
            'Correlacion con Lag': 'dashboards/static/correlation_lag.png',
            'Distribucion de Segmentos': 'dashboards/static/segments_distribution.png',
            'PCA de Segmentos': 'dashboards/static/segments_pca.png',
            'Perfiles RFM de Segmentos': 'dashboards/static/segments_rfm_profiles.png'
        }
        
        cols = st.columns(2)
        for idx, (title, path) in enumerate(images.items()):
            with cols[idx % 2]:
                if Path(path).exists():
                    st.subheader(title)
                    img = Image.open(path)
                    st.image(img, use_container_width=True)
                else:
                    st.info(f"{title} no generado aun")
        
        st.markdown("---")
        
        st.subheader("Resultados de Correlacion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr = load_data('data/processed/correlation_general.csv')
            if corr is not None:
                st.write("**Correlacion General**")
                st.dataframe(corr, use_container_width=True, hide_index=True)
        
        with col2:
            corr_seg = load_data('data/processed/correlation_by_segment.csv')
            if corr_seg is not None:
                st.write("**Por Segmento**")
                st.dataframe(corr_seg, use_container_width=True, hide_index=True)
        
        corr_prod = load_data('data/processed/correlation_by_product.csv')
        if corr_prod is not None:
            st.subheader("Top Productos por Correlacion")
            top_10 = corr_prod.nlargest(10, 'correlation')
            
            fig = px.bar(top_10, x='correlation', y='product',
                        orientation='h',
                        color='correlation',
                        color_continuous_scale='Viridis',
                        title='Top 10 Productos')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # PESTAÑA 4: REGISTROS
    # ==========================================
    with tab4:
        st.header("Registros del Pipeline")
        
        log_path = Path('pipeline_execution.log')
        
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                logs = f.readlines()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_errors = st.checkbox("Mostrar ERROR", value=True)
            with col2:
                show_warnings = st.checkbox("Mostrar WARNING", value=True)
            with col3:
                show_info = st.checkbox("Mostrar INFO", value=True)
            with col4:
                num_lines = st.selectbox("Lineas", [50, 100, 200, 500], index=0)
            
            filtered_logs = []
            for line in logs[-num_lines:]:
                if (show_errors and 'ERROR' in line) or \
                   (show_warnings and 'WARNING' in line) or \
                   (show_info and 'INFO' in line):
                    filtered_logs.append(line)
            
            st.text_area("Registros", "".join(filtered_logs), height=400)
            
            st.markdown("---")
            st.subheader("Estadisticas de Registros")
            
            col1, col2, col3 = st.columns(3)
            
            error_count = sum(1 for line in logs if 'ERROR' in line)
            warning_count = sum(1 for line in logs if 'WARNING' in line)
            info_count = sum(1 for line in logs if 'INFO' in line)
            
            with col1:
                st.metric("Errores", error_count, delta=-error_count if error_count > 0 else None, delta_color="inverse")
            with col2:
                st.metric("Advertencias", warning_count)
            with col3:
                st.metric("Informacion", info_count)
        else:
            st.warning("No se encontro archivo de registro en pipeline_execution.log")
    
    # ==========================================
    # PESTAÑA 5: HISTORIAL
    # ==========================================
    with tab5:
        st.header("Historial de Ejecuciones")
        
        history = get_execution_history()
        
        if history:
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            df_history = df_history.sort_values('timestamp', ascending=False)
            
            st.dataframe(
                df_history.style.apply(
                    lambda row: ['background-color: #d4edda' if row['status'] == 'success' 
                                else 'background-color: #f8d7da' if row['status'] == 'error'
                                else 'background-color: #fff3cd' for _ in row],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Tendencia de exito
            st.subheader("Tendencia de Exito")
            
            df_history['success'] = (df_history['status'] == 'success').astype(int)
            df_history['cumulative_success_rate'] = df_history['success'].expanding().mean() * 100
            
            fig = px.line(df_history, x='timestamp', y='cumulative_success_rate',
                         title='Tasa de Exito Acumulada',
                         markers=True)
            fig.update_layout(yaxis_title='Tasa de Exito (%)', yaxis_range=[0, 105])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay historial de ejecuciones. Ejecute el pipeline para ver el historial!")
    
    # ==========================================
    # BARRA LATERAL
    # ==========================================
    with st.sidebar:
        st.header("Configuracion")
        
        auto_refresh = st.checkbox("Auto-actualizar (10s)", value=False)
        
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Acciones Rapidas")
        
        if st.button("Ejecutar Pipeline Completo", use_container_width=True, type="primary"):
            run_pipeline()
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        
        if st.button("Limpiar Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache limpiado!")
        
        st.markdown("---")
        
        st.subheader("Alertas del Sistema")
        
        # Estado del pipeline
        files = {
            'ventas': 'data/processed/sales_processed.csv',
            'redes_sociales': 'data/processed/social_media_merged_processed.csv',
            'agregados_diarios': 'data/processed/daily_aggregates_processed.csv',
            'segmentos': 'data/processed/customer_segments.csv',
            'correlacion': 'data/processed/correlation_general.csv'
        }
        
        exists = sum(1 for f in files.values() if Path(f).exists())
        total = len(files)
        if exists < total:
            st.warning(f"{total - exists} datasets faltantes")
        
        history = get_execution_history()
        if history:
            last = history[-1]
            if last['status'] == 'success':
                st.success("Ultima ejecucion: EXITO")
            else:
                st.error(f"Ultima ejecucion: {last['status'].upper()}")

if __name__ == '__main__':
    main()