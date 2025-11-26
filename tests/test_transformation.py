"""
Tests para el módulo de transformación
Verifican la lógica de limpieza y transformación
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../src')
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_transformation import DataTransformation


class TestDataTransformation:
    """Tests para transformaciones de datos"""
    
    def setup_method(self):
        """Se ejecuta antes de cada test"""
        self.transformer = DataTransformation()
    
    # =========================================================
    # TESTS DE INFERENCIA DE CATEGORÍAS
    # =========================================================
    
    def test_infer_category_protein(self):
        """Test: Inferir categoría de proteínas"""
        assert self.transformer._infer_category("Whey Protein") == "Protein"
        assert self.transformer._infer_category("whey isolate") == "Protein"
        assert self.transformer._infer_category("CASEIN POWDER") == "Protein"
    
    def test_infer_category_amino_acids(self):
        """Test: Inferir categoría de aminoácidos"""
        assert self.transformer._infer_category("BCAA") == "Amino Acids"
        assert self.transformer._infer_category("Creatine Monohydrate") == "Amino Acids"
        assert self.transformer._infer_category("L-Glutamine") == "Amino Acids"
    
    def test_infer_category_vitamins(self):
        """Test: Inferir categoría de vitaminas"""
        assert self.transformer._infer_category("Vitamin D3") == "Vitamins"
        assert self.transformer._infer_category("Multivitamin Pack") == "Vitamins"
        assert self.transformer._infer_category("Omega-3 Fish Oil") == "Vitamins"
    
    def test_infer_category_performance(self):
        """Test: Inferir categoría de performance"""
        assert self.transformer._infer_category("Pre-Workout") == "Performance"
        assert self.transformer._infer_category("Energy Booster") == "Performance"
        assert self.transformer._infer_category("Caffeine Pills") == "Performance"
    
    
    def test_infer_category_unknown(self):
        """Test: Categoría desconocida retorna 'Other'"""
        assert self.transformer._infer_category("Random Product") == "Other"
        assert self.transformer._infer_category("") == "Other"
        assert self.transformer._infer_category(None) == "Other"
    
    # =========================================================
    # TESTS DE TRANSFORMACIÓN DE VENTAS
    # =========================================================
    
    def test_transform_sales_creates_ids(self):
        """Test: Transformación crea IDs necesarios"""
        # Crear datos de prueba
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'product_name': ['Whey Protein'] * 10,
            'quantity': [1, 2, 1, 3, 1, 2, 1, 1, 2, 1],
            'unit_price': [45.99] * 10,
            'location': ['New York'] * 10,
            'platform': ['Online'] * 10
        })
        
        result = self.transformer.transform_sales_data(df)
        
        # Verificar que se crearon los IDs
        assert 'transaction_id' in result.columns
        assert 'product_id' in result.columns
        assert 'customer_id' in result.columns
        
        # Verificar unicidad de transaction_id
        assert result['transaction_id'].is_unique
    
    def test_transform_sales_calculates_amount(self):
        """Test: Calcula amount correctamente"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'product_name': ['Product A'] * 5,
            'quantity': [1, 2, 3, 4, 5],
            'unit_price': [10.0, 10.0, 10.0, 10.0, 10.0],
            'location': ['NY'] * 5
        })
        
        result = self.transformer.transform_sales_data(df)
        
        assert 'amount' in result.columns
        expected_amounts = [10.0, 20.0, 30.0, 40.0, 50.0]
        pd.testing.assert_series_equal(
            result['amount'],
            pd.Series(expected_amounts, name='amount'),
            check_dtype=False
        )
    
    def test_transform_sales_creates_temporal_features(self):
        """Test: Crea features temporales"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'product_name': ['Product'] * 10,
            'quantity': [1] * 10,
            'unit_price': [10.0] * 10,
            'location': ['NY'] * 10
        })
        
        result = self.transformer.transform_sales_data(df)
        
        # Verificar features temporales
        temporal_features = ['year', 'month', 'quarter', 'day_of_week', 
                            'day_name', 'week', 'is_weekend']
        for feature in temporal_features:
            assert feature in result.columns, f"Feature '{feature}' faltante"
    
    def test_transform_sales_removes_invalid_values(self):
        """Test: Elimina valores inválidos"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'product_name': ['Product'] * 10,
            'quantity': [1, -1, 2, 0, 3, 1, 2, -5, 1, 2],  # Algunos negativos/cero
            'unit_price': [10.0] * 10,
            'location': ['NY'] * 10
        })
        
        result = self.transformer.transform_sales_data(df)
        
        # No debe haber cantidades negativas o cero
        assert (result['quantity'] > 0).all()
        # El resultado debe tener menos filas
        assert len(result) < len(df)
    
    # =========================================================
    # TESTS DE SOCIAL MEDIA
    # =========================================================
    def test_engagement_instagram_calculates_rate(self):
         """Test que la transformación de Instagram calcula engagement rate"""
         transformer = DataTransformation()
    
         df = pd.DataFrame({
            "likes": [100],
            "comments": [20],
            "shares": [10],
            "followers": [1000],
            "date": pd.to_datetime(["2023-01-01"])
         })

         result = transformer.transform_social_media_data(df, 'instagram')
    
         # Verificar que se calculó engagement_rate
         assert 'engagement_rate' in result.columns
         expected_rate = (100 + 20 + 10) / 1000  # 0.13
         assert abs(result['engagement_rate'].iloc[0] - expected_rate) < 0.001
         assert 'platform' in result.columns
         assert result['platform'].iloc[0] == 'instagram'


    def test_merge_social_media_data(self):
        """Test: Combina múltiples plataformas"""
        ig_df = pd.DataFrame({
            'post_id': ['IG001'],
            'date': pd.date_range('2024-01-01', periods=1),
            'platform': ['instagram'],
            'engagement_rate': [0.05]
        })
        
        tt_df = pd.DataFrame({
            'video_id': ['TT001'],
            'date': pd.date_range('2024-01-02', periods=1),
            'platform': ['tiktok'],
            'engagement_rate': [0.08]
        })
        
        yt_df = pd.DataFrame({
            'video_id': ['YT001'],
            'date': pd.date_range('2024-01-03', periods=1),
            'platform': ['youtube'],
            'engagement_rate': [0.03]
        })
        
        result = self.transformer.merge_social_media_data(ig_df, tt_df, yt_df)
        
        assert len(result) == 3
        assert set(result['platform'].unique()) == {'instagram', 'tiktok', 'youtube'}
    
    # =========================================================
    # TESTS DE AGREGADOS DIARIOS - CORREGIDO
    # =========================================================
    
    def test_calculate_daily_aggregates(self):
        """Test: Calcula agregados diarios correctamente"""
        # Datos de ventas
        sales_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'customer_id': [1, 2, 1],
            'product_id': [1, 2, 1],
            'quantity': [1, 2, 1],
            'amount': [10.0, 20.0, 15.0]
        })
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Datos de social media CON likes y comments
        social_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'platform': ['instagram', 'tiktok'],
            'engagement_rate': [0.05, 0.08],
            'likes': [100, 200],  # AÑADIDO
            'comments': [10, 20]   # AÑADIDO
        })
        social_df['date'] = pd.to_datetime(social_df['date'])
        
        result = self.transformer.calculate_daily_aggregates(sales_df, social_df)
        
        # Verificar estructura básica
        assert 'date' in result.columns
        assert 'total_sales' in result.columns
        assert 'num_transactions' in result.columns
        assert 'avg_engagement' in result.columns
        
        # Verificar columnas de interacciones (agregadas de social media)
        assert 'total_likes' in result.columns
        assert 'total_comments' in result.columns
        
        # Verificar cálculos del primer día
        day1 = result[result['date'] == '2024-01-01'].iloc[0]
        assert day1['total_sales'] == 30.0  # 10 + 20
        assert day1['num_transactions'] == 2
        assert day1['avg_engagement'] == 0.05
        assert day1['total_likes'] == 100
        assert day1['total_comments'] == 10
    
    # =========================================================
    # TESTS DE OUTLIERS
    # =========================================================
    
    def test_remove_outliers_iqr(self):
        """Test: Elimina outliers correctamente"""
        df = pd.DataFrame({
            'amount': [10, 12, 11, 13, 10, 12, 100, 11, 10, 12]  # 100 es outlier
        })
        
        result = self.transformer._remove_outliers(df, 'amount', method='iqr')
        
        # El outlier (100) debe ser eliminado
        assert len(result) < len(df)
        assert 100 not in result['amount'].values


# =========================================================
# TESTS DE INTEGRACIÓN
# =========================================================

class TestTransformationIntegration:
    """Tests de integración del módulo completo"""
    
    def test_full_transformation_pipeline(self):
        """Test: Pipeline completo de transformación"""
        transformer = DataTransformation()
        
        # Datos de prueba
        data = {
            'sales': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100),
                'product_name': ['Whey Protein'] * 100,
                'quantity': np.random.randint(1, 5, 100),
                'unit_price': np.random.uniform(20, 50, 100),
                'location': ['NY'] * 100,
                'platform': ['Online'] * 100
            }),
            'instagram': pd.DataFrame({
                'post_id': [f'IG{i}' for i in range(50)],
                'date': pd.date_range('2024-01-01', periods=50),
                'influencer': ['user1'] * 50,
                'followers': [10000] * 50,
                'likes': np.random.randint(500, 5000, 50),
                'comments': np.random.randint(50, 500, 50),
                'shares': np.random.randint(10, 100, 50)
            }),
            'tiktok': None,
            'youtube': None
        }
        
        # Ejecutar transformación completa
        result = transformer.transform_all(data)
        
        # Verificar que se generaron los datasets esperados
        assert 'sales' in result
        assert 'instagram' in result
        assert 'social_media_merged' in result
        assert 'daily_aggregates' in result
        
        # Verificar integridad
        assert len(result['sales']) > 0
        assert len(result['daily_aggregates']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])