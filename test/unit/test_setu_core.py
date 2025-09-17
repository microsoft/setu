"""
Unit tests for setu core functionality.
"""

import pytest


class TestNativeExtension:
    """Test the native C++ extension if available."""

    @pytest.fixture
    def processor(self):
        """Create a native processor if available."""
        try:
            from setu._native import CoreProcessor

            return CoreProcessor()
        except ImportError:
            pytest.skip("Native extension not available")

    @pytest.mark.unit
    def test_native_processor_initialization(self, processor) -> None:
        """Test native processor initialization."""
        assert processor is not None

    @pytest.mark.unit
    def test_native_process_functionality(self, processor) -> None:
        """Test native processor functionality."""
        input_data = ["native", "test"]
        result = processor.process(input_data)

        assert len(result) == 2
        assert result[0] == "processed: native"
        assert result[1] == "processed: test"

    @pytest.mark.unit
    def test_native_process_counter(self, processor) -> None:
        """Test native processor counter functionality."""
        initial_count = processor.get_processed_count()

        processor.process(["test1", "test2", "test3"])

        new_count = processor.get_processed_count()
        assert new_count == initial_count + 3

    @pytest.mark.unit
    def test_native_empty_input(self, processor) -> None:
        """Test native processor with empty input."""
        initial_count = processor.get_processed_count()
        result = processor.process([])

        assert result == []
        assert processor.get_processed_count() == initial_count

    @pytest.mark.unit
    def test_native_large_batch(self, processor) -> None:
        """Test native processor with large batch."""
        input_data = [f"native_item_{i}" for i in range(100)]
        result = processor.process(input_data)

        assert len(result) == 100
        assert all("processed: native_item_" in item for item in result)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for SetuCore."""


@pytest.mark.integration
class TestIntegration:
    """Integration tests for SetuCore."""
