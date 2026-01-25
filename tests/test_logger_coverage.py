# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_logger_dir_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that logger creation logic (module level) creates the directory if it doesn't exist.
    """
    # 1. Unload the module if it's already loaded so we can re-import it
    if "coreason_ai_gateway.utils.logger" in sys.modules:
        del sys.modules["coreason_ai_gateway.utils.logger"]

    # 2. Mock pathlib.Path
    mock_path_instance = MagicMock()

    # Configure the mock to return False for exists() initially
    mock_path_instance.exists.return_value = False

    # Let's use `unittest.mock.patch` on `pathlib.Path`?
    # Since `logger.py` does `from pathlib import Path`, we need to patch `pathlib.Path`.
    # Pytest runner relies on Path, so we must be careful.

    # Let's try patching it only during import.
    with patch("pathlib.Path") as mock_path:
        mock_instance = mock_path.return_value
        mock_instance.exists.return_value = False  # Force "not exists"

        # Re-import
        import coreason_ai_gateway.utils.logger  # noqa: F401

        # Verify mkdir was called
        # The code calls: log_path.mkdir(parents=True, exist_ok=True)
        # log_path is Path("logs") -> mock_instance
        mock_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Cleanup: Restore sys.modules to real module?
    # If we leave the mocked module in sys.modules, other tests might fail.
    # So we should remove it from sys.modules so subsequent imports get the real one.
    if "coreason_ai_gateway.utils.logger" in sys.modules:
        del sys.modules["coreason_ai_gateway.utils.logger"]

    # Re-import real module to restore state for other tests?
    import coreason_ai_gateway.utils.logger  # noqa: F401
