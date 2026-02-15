"""
Test: Verify Telegram Bridge can be initialized
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

def test_telegram_bridge_init():
    """Test that TelegramBridge can be initialized with a token."""
    with patch('med_safety_gym.telegram_bridge.Application') as mock_app_class:
        # Mock the application builder chain
        mock_builder = MagicMock()
        mock_app = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.build.return_value = mock_app
        mock_app_class.builder.return_value = mock_builder
        
        from med_safety_gym.telegram_bridge import TelegramBridge
        
        bridge = TelegramBridge("test-token-123")
        
        assert bridge.token == "test-token-123"
        assert bridge.agent is not None
        mock_app_class.builder.assert_called_once()
        mock_builder.token.assert_called_once_with("test-token-123")

if __name__ == "__main__":
    test_telegram_bridge_init()
    print("✅ Telegram bridge initialization test passed!")
