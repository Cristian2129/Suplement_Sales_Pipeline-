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

st.set_page_config(
    page_title="Pipeline Supplement Sales",
    page_icon="💊",
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
# EJECUTOR DE PIPELINE
# ==========================================

def run_pipeline():
    """
    Ejecuta el pipeline completo
    """
    import sys
    import os
    
    # Agregar directorio src al path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    
    try:
        start_time = time.time()
        
        # Importar orchestrator
        from orchestrator import DataPipelineOrchestrator
        
        # Crear orchestrator y ejecutar
        with st.spinner('Ejecutando pipeline...'):
            orchestrator = DataPipelineOrchestrator()
            success = orchestrator.run_complete_pipeline()
        
        duration = time.time() - start_time
        
        if success:
            save_execution_history('success', duration)
            st.success(f'Pipeline completado en {duration:.1f}s')
            return True
        else:
            save_execution_history('failed', duration, 'Pipeline returned False')
            st.error('Pipeline fallo')
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        save_execution_history('error', duration, str(e))
        st.error(f'Error ejecutando pipeline: {e}')
        return False


# ==========================================
# DASHBOARD PRINCIPAL
# ==========================================

def main():
    # Header mejorado
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.title("Pipeline Observatory")
        st.caption("Observabilidad completa del pipeline de datos")
    with col2:
        if st.button("Refresh", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        if st.button("Run Pipeline", use_container_width=True, type="primary"):
            run_pipeline()
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # ==========================================
    # TABS PRINCIPALES
    # ==========================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Infrastructure", 
        "Performance",
        "Analysis Results",
        "Logs",
        "History"
    ])
    
    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab1:
        st.header("Pipeline Status Overview")
        
        # Estado de archivos
        files = {
            'sales': 'data/processed/sales_processed.csv',
            'social': 'data/processed/social_media_merged_processed.csv',
            'daily': 'data/processed/daily_aggregates_processed.csv',
            'segments': 'data/processed/customer_segments.csv',
            'correlation': 'data/processed/correlation_general.csv'
        }
        
        total = len(files)
        exists = sum(1 for f in files.values() if Path(f).exists())
        
        # Metricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status = "OK" if exists == total else "Partial"
            st.metric("Pipeline Status", status, f"{exists}/{total} datasets")
        
        with col2:
            sales = load_data(files['sales'])
            records = len(sales) if sales is not None else 0
            st.metric("Total Records", f"{records:,}")
        
        with col3:
            daily = load_data(files['daily'])
            days = len(daily) if daily is not None else 0
            st.metric("Days Analyzed", f"{days:,}")
        
        with col4:
            corr = load_data(files['correlation'])
            if corr is not None and len(corr) > 0:
                r = corr['r'].iloc[0]
                st.metric("Correlation", f"{r:.3f}")
            else:
                st.metric("Correlation", "N/A")
        
        with col5:
            # Ultima ejecucion
            history = get_execution_history()
            if history:
                last = history[-1]
                last_time = datetime.fromisoformat(last['timestamp'])
                time_ago = (datetime.now() - last_time).seconds // 60
                st.metric("Last Run", f"{time_ago}m ago", 
                         "✅" if last['status'] == 'success' else "❌")
            else:
                st.metric("Last Run", "Never", "➖")
        
        st.markdown("---")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Data Summary")
            
            if sales is not None:
                st.write(f"**Sales**: {len(sales):,} transactions")
                if 'amount' in sales.columns:
                    total_revenue = sales['amount'].sum()
                    st.write(f"**Revenue**: ${total_revenue:,.0f}")
                if 'customer_id' in sales.columns:
                    st.write(f"**Customers**: {sales['customer_id'].nunique():,}")
                if 'product_name' in sales.columns:
                    st.write(f"**Products**: {sales['product_name'].nunique()}")
            
            social = load_data(files['social'])
            if social is not None:
                st.write(f"**Social Posts**: {len(social):,}")
                if 'platform' in social.columns:
                    platforms = ', '.join(social['platform'].unique())
                    st.write(f"**Platforms**: {platforms}")
        
        with col2:
            st.subheader("Key Insights")
            
            if corr is not None and len(corr) > 0:
                r = corr['r'].iloc[0]
                sig = "Significant" if corr['significant'].iloc[0] else "Not significant"
                direction = "Positive" if r > 0 else "Negative"
                st.write(f"**Correlation**: {r:.3f} ({direction})")
                st.write(f"**Significance**: {sig}")
            
            segments = load_data(files['segments'])
            if segments is not None and 'segment_name' in segments.columns:
                top_segment = segments['segment_name'].value_counts().index[0]
                count = segments['segment_name'].value_counts().iloc[0]
                st.write(f"**Largest Segment**: {top_segment} ({count})")
            
            lag = load_data('data/processed/correlation_lag.csv')
            if lag is not None and len(lag) > 0:
                best = lag.loc[lag['correlation'].abs().idxmax()]
                st.write(f"**Optimal Lag**: {best['lag_days']:.0f} days")
        
        with col3:
            st.subheader("File Status")
            
            for name, path in files.items():
                if Path(path).exists():
                    size = Path(path).stat().st_size / 1024
                    modified = datetime.fromtimestamp(Path(path).stat().st_mtime)
                    time_ago = datetime.now() - modified
                    
                    if time_ago.seconds < 3600:
                        time_str = f"{time_ago.seconds // 60}m ago"
                    else:
                        time_str = f"{time_ago.seconds // 3600}h ago"
                    
                    st.write(f"**{name}**: {size:.0f}KB ({time_str})")
                else:
                    st.write(f"**{name}**: Missing")
    
    # ==========================================
    # TAB 2: INFRASTRUCTURE (igual que antes)
    # ==========================================
    with tab2:
        st.header("Infrastructure Metrics")
        
        sys_metrics = get_system_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{sys_metrics['cpu_percent']:.1f}%")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_metrics['cpu_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Memory Usage", 
                     f"{sys_metrics['memory_percent']:.1f}%",
                     delta=f"{sys_metrics['memory_used_gb']:.1f}GB / {sys_metrics['memory_total_gb']:.1f}GB")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_metrics['memory_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RAM"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgreen"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.metric("Disk Usage", 
                     f"{sys_metrics['disk_percent']:.1f}%",
                     delta=f"{sys_metrics['disk_used_gb']:.0f}GB / {sys_metrics['disk_total_gb']:.0f}GB")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_metrics['disk_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Disk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgreen"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**OS**: {psutil.os.name}")
            st.write(f"**CPU Cores**: {sys_metrics['cpu_count']}")
            st.write(f"**Total RAM**: {sys_metrics['memory_total_gb']:.1f} GB")
            st.write(f"**Total Disk**: {sys_metrics['disk_total_gb']:.0f} GB")
        
        with col2:
            process = psutil.Process()
            st.write(f"**Python Memory**: {process.memory_info().rss / (1024**2):.0f} MB")
            st.write(f"**Threads**: {process.num_threads()}")
            st.write(f"**Uptime**: {(time.time() - process.create_time()) / 3600:.1f}h")
    
    # ==========================================
    # TAB 3: PERFORMANCE
    # ==========================================
    with tab3:
        st.header("Performance Metrics")
        
        history = get_execution_history()
        
        if history:
            # Convertir a DataFrame
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            df_history = df_history.sort_values('timestamp')
            
            # Grafico de tiempos de ejecucion
            st.subheader("Execution Times")
            
            fig = px.line(df_history, x='timestamp', y='duration_seconds',
                         title='Pipeline Execution Duration Over Time',
                         markers=True,
                         color='status',
                         color_discrete_map={'success': 'green', 'failed': 'red', 'error': 'orange'})
            fig.update_layout(xaxis_title='Date', yaxis_title='Duration (seconds)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadisticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_duration = df_history['duration_seconds'].mean()
                st.metric("Avg Duration", f"{avg_duration:.1f}s")
            
            with col2:
                min_duration = df_history['duration_seconds'].min()
                st.metric("Fastest", f"{min_duration:.1f}s")
            
            with col3:
                max_duration = df_history['duration_seconds'].max()
                st.metric("Slowest", f"{max_duration:.1f}s")
            
            with col4:
                success_rate = (df_history['status'] == 'success').sum() / len(df_history) * 100
                st.metric("Success Rate", f"{success_rate:.0f}%")
        else:
            st.info("No execution history available. Run the pipeline first!")
        
        st.markdown("---")
        
        # Tamanos de archivos
        st.subheader("File Sizes")
        
        files = {
            'sales': 'data/processed/sales_processed.csv',
            'social': 'data/processed/social_media_merged_processed.csv',
            'daily': 'data/processed/daily_aggregates_processed.csv',
            'segments': 'data/processed/customer_segments.csv'
        }
        
        file_data = []
        for name, path in files.items():
            if Path(path).exists():
                size_kb = Path(path).stat().st_size / 1024
                file_data.append({'Dataset': name, 'Size (KB)': size_kb})
        
        if file_data:
            df_sizes = pd.DataFrame(file_data)
            fig = px.bar(df_sizes, x='Dataset', y='Size (KB)',
                        color='Size (KB)',
                        color_continuous_scale='Blues',
                        title='Dataset Sizes')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # TAB 4: ANALYSIS RESULTS (igual que antes)
    # ==========================================
    with tab4:
        st.header("Analysis Results")
        
        st.subheader("Visualizations")
        
        images = {
            'Correlation Scatter': 'dashboards/static/correlation_scatter.png',
            'Correlation by Segment': 'dashboards/static/correlation_by_segment.png',
            'Correlation Lag': 'dashboards/static/correlation_lag.png',
            'Segments Distribution': 'dashboards/static/segments_distribution.png',
            'Segments PCA': 'dashboards/static/segments_pca.png',
            'Segments RFM Profiles': 'dashboards/static/segments_rfm_profiles.png'
        }
        
        cols = st.columns(2)
        for idx, (title, path) in enumerate(images.items()):
            with cols[idx % 2]:
                if Path(path).exists():
                    st.subheader(title)
                    img = Image.open(path)
                    st.image(img, use_container_width=True)
                else:
                    st.info(f"{title} not generated yet")
        
        st.markdown("---")
        
        st.subheader("Correlation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr = load_data('data/processed/correlation_general.csv')
            if corr is not None:
                st.write("**General Correlation**")
                st.dataframe(corr, use_container_width=True, hide_index=True)
        
        with col2:
            corr_seg = load_data('data/processed/correlation_by_segment.csv')
            if corr_seg is not None:
                st.write("**By Segment**")
                st.dataframe(corr_seg, use_container_width=True, hide_index=True)
        
        corr_prod = load_data('data/processed/correlation_by_product.csv')
        if corr_prod is not None:
            st.subheader("Top Products by Correlation")
            top_10 = corr_prod.nlargest(10, 'correlation')
            
            fig = px.bar(top_10, x='correlation', y='product',
                        orientation='h',
                        color='correlation',
                        color_continuous_scale='Viridis',
                        title='Top 10 Products')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # TAB 5: LOGS (igual que antes)
    # ==========================================
    with tab5:
        st.header("Pipeline Logs")
        
        log_path = Path('pipeline_execution.log')
        
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                logs = f.readlines()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_errors = st.checkbox("Show ERROR", value=True)
            with col2:
                show_warnings = st.checkbox("Show WARNING", value=True)
            with col3:
                show_info = st.checkbox("Show INFO", value=True)
            with col4:
                num_lines = st.selectbox("Lines", [50, 100, 200, 500], index=0)
            
            filtered_logs = []
            for line in logs[-num_lines:]:
                if (show_errors and 'ERROR' in line) or \
                   (show_warnings and 'WARNING' in line) or \
                   (show_info and 'INFO' in line):
                    filtered_logs.append(line)
            
            st.text_area("Logs", "".join(filtered_logs), height=400)
            
            st.markdown("---")
            st.subheader("Log Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            error_count = sum(1 for line in logs if 'ERROR' in line)
            warning_count = sum(1 for line in logs if 'WARNING' in line)
            info_count = sum(1 for line in logs if 'INFO' in line)
            
            with col1:
                st.metric("Errors", error_count, delta=-error_count if error_count > 0 else None, delta_color="inverse")
            with col2:
                st.metric("Warnings", warning_count)
            with col3:
                st.metric("Info", info_count)
        else:
            st.warning("No log file found at pipeline_execution.log")
    
    # ==========================================
    # TAB 6: HISTORY (NUEVA)
    # ==========================================
    with tab6:
        st.header("Execution History")
        
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
            st.subheader("Success Trend")
            
            df_history['success'] = (df_history['status'] == 'success').astype(int)
            df_history['cumulative_success_rate'] = df_history['success'].expanding().mean() * 100
            
            fig = px.line(df_history, x='timestamp', y='cumulative_success_rate',
                         title='Cumulative Success Rate',
                         markers=True)
            fig.update_layout(yaxis_title='Success Rate (%)', yaxis_range=[0, 105])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No execution history. Run the pipeline to see history!")
    
    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        st.header("Settings")
        
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
        
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Quick Actions")
        
        if st.button("Run Full Pipeline", use_container_width=True, type="primary"):
            run_pipeline()
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        st.markdown("---")
        
        st.subheader("System Alerts")
        
        sys = get_system_metrics()
        
        if sys['cpu_percent'] > 80:
            st.error("High CPU usage!")
        
        if sys['memory_percent'] > 85:
            st.error("High memory usage!")
        
        if sys['disk_percent'] > 90:
            st.error("Disk almost full!")
        
        exists = sum(1 for f in files.values() if Path(f).exists())
        total = len(files)
        if exists < total:
            st.warning(f"{total - exists} datasets missing")
        
        # Estado del pipeline
        history = get_execution_history()
        if history:
            last = history[-1]
            if last['status'] == 'success':
                st.success("Last run: SUCCESS")
            else:
                st.error(f"Last run: {last['status'].upper()}")

if __name__ == '__main__':
    import os
    main()