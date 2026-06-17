"""Pytest configuration and global fixture imports."""

# Register mock_npu fixture as a plugin so it is auto-loaded in unit/integration tests
pytest_plugins = ["tests.fixtures.mock_npu"]
