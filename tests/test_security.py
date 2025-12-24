
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from med_safety_gym.evaluation_service import EvaluationManager
from med_safety_gym.dipg_environment import DIPGEnvironment

class TestSecurity:
    """Security tests for the evaluation service."""
    
    @pytest.fixture
    def eval_manager(self):
        env = Mock(spec=DIPGEnvironment)
        return EvaluationManager(env)
        
    def test_save_path_traversal_prevention(self, eval_manager):
        """Test that path traversal attempts raise ValueError."""
        # Mock data
        detailed_results = []
        summary = Mock()
        summary.model_dump.return_value = {}
        
        # Test cases for path traversal
        unsafe_paths = [
            "../../etc/passwd",
            "../outside.json",
            "/tmp/absolute_path.json",
            "subdir/../../../../root.json"
        ]
        
        for path in unsafe_paths:
            with pytest.raises(ValueError, match="Invalid save_path"):
                eval_manager._save_results(detailed_results, summary, path)
                
    def test_save_path_valid_paths(self, eval_manager, tmp_path):
        """Test that valid relative paths are allowed."""
        # Mock data
        detailed_results = []
        summary = Mock()
        summary.model_dump.return_value = {}
        
        # We need to mock Path.cwd() to return tmp_path so we can write safely during test
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            # 1. Simple filename
            path1 = "results.json"
            saved_path1 = eval_manager._save_results(detailed_results, summary, path1)
            assert Path(saved_path1).name == "results.json"
            
            # 2. Subdirectory
            path2 = "evals/run1.json"
            saved_path2 = eval_manager._save_results(detailed_results, summary, path2)
            assert Path(saved_path2).name == "run1.json"
            assert "evals" in str(saved_path2)
