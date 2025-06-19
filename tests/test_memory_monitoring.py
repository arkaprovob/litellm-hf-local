"""
Test memory monitoring functionality.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from src.hf_local_adapter.utils.memory import (
    get_gpu_memory_info,
    print_gpu_memory_usage,
    print_model_device_map,
    print_model_memory_footprint,
    get_memory_footprint_estimate
)


class TestMemoryMonitoring:
    """Test memory monitoring utilities."""
    
    def test_get_gpu_memory_info_no_cuda(self):
        """Test GPU memory info when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            result = get_gpu_memory_info()
            assert result == {}
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_get_gpu_memory_info_with_cuda(self, mock_props, mock_reserved, mock_allocated, mock_count, mock_available):
        """Test GPU memory info when CUDA is available."""
        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 24 * 1024**3  # 24GB
        mock_props.return_value = mock_device
        
        # Mock memory usage
        mock_allocated.side_effect = [8 * 1024**3, 4 * 1024**3]  # 8GB, 4GB
        mock_reserved.side_effect = [10 * 1024**3, 6 * 1024**3]  # 10GB, 6GB
        
        result = get_gpu_memory_info()
        
        assert len(result) == 2
        assert result[0]['allocated'] == 8.0
        assert result[0]['reserved'] == 10.0
        assert result[0]['total'] == 24.0
        assert result[0]['free'] == 14.0
        
        assert result[1]['allocated'] == 4.0
        assert result[1]['reserved'] == 6.0
        assert result[1]['total'] == 24.0
        assert result[1]['free'] == 18.0
    
    def test_get_memory_footprint_estimate(self):
        """Test memory footprint estimation."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.Linear(50, 10)
        )
        
        footprint = get_memory_footprint_estimate(model)
        
        # Linear(100, 50) has 100*50 + 50 = 5050 parameters
        # Linear(50, 10) has 50*10 + 10 = 510 parameters
        # Total: 5560 parameters
        assert footprint['total_parameters'] == 5560
        assert footprint['trainable_parameters'] == 5560
        assert footprint['parameter_memory_gb'] > 0
        assert footprint['estimated_inference_memory_gb'] > footprint['parameter_memory_gb']
        assert footprint['estimated_training_memory_gb'] > footprint['estimated_inference_memory_gb']
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_print_gpu_memory_usage_no_cuda(self, mock_available, caplog):
        """Test GPU memory usage printing when CUDA is not available."""
        print_gpu_memory_usage("Test Title")
        assert "CUDA not available" in caplog.text
    
    def test_print_model_device_map_with_hf_device_map(self, caplog):
        """Test printing model device map with HuggingFace device map."""
        # Create a mock model with hf_device_map
        model = MagicMock()
        model.hf_device_map = {
            'layer1': 0,
            'layer2': 1,
            'layer3': 'cpu'
        }
        
        print_model_device_map(model, "Test Model")
        
        # Check that device map is logged
        assert "Test Model" in caplog.text
        assert "HuggingFace Device Map" in caplog.text
    
    def test_print_model_memory_footprint(self, caplog):
        """Test printing model memory footprint."""
        # Create a simple model
        model = torch.nn.Linear(100, 50)
        
        print_model_memory_footprint(model, "Test Model")
        
        # Check that memory info is logged
        assert "Test Model" in caplog.text
        assert "Total Parameters" in caplog.text
        assert "Parameter Memory" in caplog.text


class TestMemoryMonitoringIntegration:
    """Integration tests for memory monitoring."""
    
    def test_memory_monitoring_with_model_config(self):
        """Test that ModelConfig correctly handles show_memory_usage parameter."""
        from src.hf_local_adapter.config.model_config import ModelConfig
        
        # Test default value
        config = ModelConfig(model_id="test/model")
        assert config.show_memory_usage is True
        
        # Test explicit False
        config = ModelConfig(model_id="test/model", show_memory_usage=False)
        assert config.show_memory_usage is False
        
        # Test explicit True
        config = ModelConfig(model_id="test/model", show_memory_usage=True)
        assert config.show_memory_usage is True


if __name__ == "__main__":
    pytest.main([__file__]) 