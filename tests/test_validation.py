"""
Tests para el módulo de validación
Verifican que las validaciones detectan problemas correctamente
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_validation import DataValidator


class TestDataValidator:
    """Tests para validaciones de datos"""
    
    def setup_method(self):
        """Se ejecuta antes de cada test"""
        self.validator = DataValidator()
    
    # =========================================================
    # TESTS DE VALIDACIONES BÁSICAS
    # =========================================================
    
    def test_validate_required_columns_pass(self):
        """Test: Validación pasa con todas las columnas"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.0, 2.0, 3.0]
        })
        
        result = self.validator._validate_required_columns(
            df, ['col1', 'col2'], 'TEST'
        )
        
        assert result == True
    
    def test_validate_required_columns_fail(self):
        """Test: Validación falla con columnas faltantes"""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        result = self.validator._validate_required_columns(
            df, ['col1', 'col2'], 'TEST'
        )
        
        assert result == False
    
    def test_validate_nulls_pass(self):
        """Test: Validación de nulos pasa sin nulos"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = self.validator._validate_nulls(
            df, ['col1', 'col2'], 'TEST'
        )
        
        assert result == True
    
    def test_validate_nulls_fail(self):
        """Test: Validación de nulos falla con nulos"""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = self.validator._validate_nulls(
            df, ['col1'], 'TEST'
        )
        
        assert result == False
    
    def test_validate_duplicates_pass(self):
        """Test: Validación de duplicados pasa sin duplicados"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c']
        })
        
        result = self.validator._validate_duplicates(
            df, ['id'], 'TEST'
        )
        
        assert result == True
    
    def test_validate_duplicates_fail(self):
        """Test: Validación de duplicados falla con duplicados"""
        df = pd.DataFrame({
            'id': [1, 2, 2],
            'name': ['a', 'b', 'c']
        })
        
        result = self.validator._validate_duplicates(
            df, ['id'], 'TEST'
        )
        
        assert result == False
    
    def test_validate_ranges_pass(self):
        """Test: Validación de rangos pasa con valores positivos"""
        df = pd.DataFrame({
            'price': [10.0, 20.0, 30.0],
            'quantity': [1, 2, 3]
        })
        
        result = self.validator._validate_ranges(
            df, ['price', 'quantity'], 'TEST', min_val=0
        )
        
        assert result == True
    
    def test_validate_ranges_fail(self):
        """Test: Validación de rangos falla con valores negativos"""
        df = pd.DataFrame({
            'price': [10.0, -20.0, 30.0],
            'quantity': [1, 2, 3]
        })
        
        result = self.validator._validate_ranges(
            df, ['price'], 'TEST', min_val=0
        )
        
        assert result == False
    
    def test_validate_dates_pass(self):
        """Test: Validación de fechas pasa con fechas válidas"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10)
        })
        
        result = self.validator._validate_dates(
            df, 'date', 'TEST'
        )
        
        assert result == True
    
    def test_validate_dates_fail_wrong_type(self):
        """Test: Validación falla si date no es datetime"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']  # Strings, no datetime
        })
        
        result = self.validator._validate_dates(
            df, 'date', 'TEST'
        )
        
        assert result == False
    
    # =========================================================
    # TESTS DE VALIDACIÓN DE VENTAS
    # =========================================================
    
    def test_validate_sales_complete_data(self):
        """Test: Validación de ventas con datos completos"""
        # Crear datos con suficientes registros para pasar validaciones
        # (min 10 customers, min 5 products)
        df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
            'product_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'product_name': ['A', 'B', 'C', 'D', 'E'] * 3,
            'date': pd.date_range('2024-01-01', periods=15),
            'quantity': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'unit_price': [10.0] * 15,
            'amount': [10.0, 20.0, 30.0] * 5,
            'transaction_id': [f'T{i:03d}' for i in range(15)]
        })
        
        result = self.validator.validate_sales(df)
        
        # Debe pasar todas las validaciones
        assert result == True
    
    def test_validate_sales_missing_columns(self):
        """Test: Validación falla con columnas faltantes"""
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'product_name': ['A', 'B', 'C']
            # Faltan: product_id, date, amount, quantity, unit_price
        })
        
        result = self.validator.validate_sales(df)
        
        # Debe fallar porque faltan columnas críticas
        assert result == False
    
    def test_validate_sales_invalid_amounts(self):
        """Test: Validación detecta amounts inválidos"""
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'product_id': [1, 2, 3],
            'product_name': ['A', 'B', 'C'],
            'date': pd.date_range('2024-01-01', periods=3),
            'quantity': [1, 2, -3],  # Cantidad negativa
            'unit_price': [10.0, 20.0, 30.0],
            'amount': [10.0, 40.0, -90.0],  # Amount negativo
            'transaction_id': ['T001', 'T002', 'T003']
        })
        
        result = self.validator.validate_sales(df)
        
        # Debe fallar por valores negativos
        assert result == False
    
    # =========================================================
    # TESTS DE VALIDACIÓN DE SOCIAL MEDIA
    # =========================================================
    
    def test_validate_social_media_instagram(self):
        """Test: Validación de Instagram"""
        df = pd.DataFrame({
            'post_id': ['IG001', 'IG002'],
            'date': pd.date_range('2024-01-01', periods=2),
            'engagement_rate': [0.05, 0.08],
            'platform': ['instagram', 'instagram']
        })
        
        result = self.validator.validate_social_media(df, 'instagram')
        
        assert result == True
    
    def test_validate_social_media_negative_engagement(self):
        """Test: Validación detecta engagement negativo"""
        df = pd.DataFrame({
            'post_id': ['IG001', 'IG002'],
            'date': pd.date_range('2024-01-01', periods=2),
            'engagement_rate': [0.05, -0.08],  # Negativo
            'platform': ['instagram', 'instagram']
        })
        
        result = self.validator.validate_social_media(df, 'instagram')
        
        assert result == False
    
    # =========================================================
    # TESTS DE VALIDACIÓN DE MERGED Y AGGREGATES
    # =========================================================
    
    def test_validate_social_media_merged(self):
        """Test: Validación de social media merged"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6),
            'platform': ['instagram', 'instagram', 'tiktok', 'tiktok', 'youtube', 'youtube'],
            'engagement_rate': [0.05, 0.06, 0.08, 0.09, 0.03, 0.04]
        })
        
        result = self.validator.validate_social_media_merged(df)
        
        assert result == True
    
    def test_validate_daily_aggregates(self):
        """Test: Validación de agregados diarios"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'total_sales': [100.0, 150.0, 200.0, 180.0, 220.0, 190.0, 210.0, 230.0, 200.0, 180.0],
            'num_transactions': [10, 15, 20, 18, 22, 19, 21, 23, 20, 18],
            'avg_engagement': [0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.07, 0.09, 0.06, 0.05],
            'num_posts': [5, 6, 7, 5, 8, 6, 7, 9, 6, 5]
        })
        
        result = self.validator.validate_daily_aggregates(df)
        
        assert result == True
    
    def test_validate_daily_aggregates_negative_sales(self):
        """Test: Validación detecta ventas negativas"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'total_sales': [100.0, -150.0, 200.0],  # Negativo
            'num_transactions': [10, 15, 20]
        })
        
        result = self.validator.validate_daily_aggregates(df)
        
        assert result == False
    
    # =========================================================
    # TESTS DE VALIDACIÓN COMPLETA
    # =========================================================
    
    def test_validate_all_with_complete_data(self):
        """Test: Validación completa con todos los datos"""
        # Crear datos con suficientes registros
        data = {
            'sales': pd.DataFrame({
                'customer_id': list(range(1, 16)),  # 15 customers
                'product_id': [1, 2, 3, 4, 5] * 3,  # 5 products
                'product_name': ['A', 'B', 'C', 'D', 'E'] * 3,
                'date': pd.date_range('2024-01-01', periods=15),
                'quantity': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                'unit_price': [10.0] * 15,
                'amount': [10.0, 20.0, 30.0] * 5,
                'transaction_id': [f'T{i:03d}' for i in range(15)]
            }),
            'instagram': pd.DataFrame({
                'post_id': ['IG001', 'IG002'],
                'date': pd.date_range('2024-01-01', periods=2),
                'engagement_rate': [0.05, 0.08],
                'platform': ['instagram', 'instagram']
            }),
            'social_media_merged': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=2),
                'platform': ['instagram', 'tiktok'],
                'engagement_rate': [0.05, 0.08]
            }),
            'daily_aggregates': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=3),
                'total_sales': [100.0, 150.0, 200.0],
                'num_transactions': [10, 15, 20]
            })
        }
        
        results = self.validator.validate_all(data)
        
        # Verificar que retorna resultados
        assert isinstance(results, dict)
        assert 'sales' in results
        assert 'instagram' in results
        
        # Sales debe pasar ahora
        assert results['sales'] == True
    
    # =========================================================
    # TESTS DE CASOS EDGE
    # =========================================================
    
    def test_validate_empty_dataframe(self):
        """Test: Validación maneja DataFrame vacío"""
        df = pd.DataFrame()
        
        result = self.validator.validate_sales(df)
        
        # Debe detectar que está vacío
        assert result == False
    
    def test_validate_none_dataframe(self):
        """Test: Validación maneja None"""
        result = self.validator.validate_sales(None)
        
        assert result == False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])