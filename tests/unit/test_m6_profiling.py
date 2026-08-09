"""Test M6: Automatic Profiling.

This test verifies:
1. ProfileResult data model
2. ProfileStore (SQLite) operations
3. ProfileExecutor measurements
4. Protobuf ProfileResponse integration
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from edgeshard.profiler.profile_data import ProfileResult
from edgeshard.profiler.store import ProfileStore


class TestProfileResult:
    """Test ProfileResult data model."""

    def test_create_profile_result(self):
        """Test creating a ProfileResult."""
        result = ProfileResult(
            model_name="Qwen/Qwen2.5-0.5B",
            model_revision="abc123",
            dtype="float16",
            device_type="cuda:0",
            device_name="NVIDIA GeForce RTX 4090",
            device_memory_mb=24564,
            layer_forward_ms=0.5,
            kv_cache_per_token_mb=0.001,
            prefill_tokens_per_sec=5000.0,
            decode_tokens_per_sec=100.0,
            total_model_memory_mb=1024.0,
            num_layers_profiled=24,
            num_runs=10,
        )

        assert result.model_name == "Qwen/Qwen2.5-0.5B"
        assert result.layer_forward_ms == 0.5
        assert result.timestamp > 0

    def test_profile_key(self):
        """Test profile_key property."""
        result = ProfileResult(
            model_name="test-model",
            model_revision="v1",
            dtype="float16",
            device_type="cuda:0",
            device_name="RTX 4090",
        )

        key = result.profile_key
        assert key == ("test-model", "v1", "float16", "cuda:0", "RTX 4090")

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ProfileResult(
            model_name="test",
            layer_forward_ms=1.5,
        )

        d = result.to_dict()
        assert d["model_name"] == "test"
        assert d["layer_forward_ms"] == 1.5
        assert "timestamp" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "model_name": "test",
            "model_revision": "",
            "dtype": "float16",
            "device_type": "cuda:0",
            "device_name": "GPU",
            "device_memory_mb": 8192,
            "layer_forward_ms": 2.0,
            "kv_cache_per_token_mb": 0.002,
            "prefill_tokens_per_sec": 3000.0,
            "decode_tokens_per_sec": 80.0,
            "total_model_memory_mb": 512.0,
            "num_layers_profiled": 12,
            "num_runs": 5,
            "timestamp": 1234567890.0,
        }

        result = ProfileResult.from_dict(data)
        assert result.model_name == "test"
        assert result.layer_forward_ms == 2.0
        assert result.num_runs == 5


class TestProfileStore:
    """Test ProfileStore SQLite operations."""

    def test_create_store(self):
        """Test creating a ProfileStore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)
            profiles = store.list_profiles()
            assert profiles == []
        finally:
            os.unlink(db_path)

    def test_save_and_lookup(self):
        """Test saving and looking up a profile."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            result = ProfileResult(
                model_name="test-model",
                model_revision="v1",
                dtype="float16",
                device_type="cuda:0",
                device_name="Test GPU",
                device_memory_mb=8192,
                layer_forward_ms=1.23,
                kv_cache_per_token_mb=0.001,
                prefill_tokens_per_sec=5000.0,
                decode_tokens_per_sec=100.0,
                total_model_memory_mb=512.0,
                num_layers_profiled=24,
            )

            store.save(result)

            # Look up
            found = store.lookup(
                model_name="test-model",
                model_revision="v1",
                dtype="float16",
                device_type="cuda:0",
                device_name="Test GPU",
            )

            assert found is not None
            assert found.model_name == "test-model"
            assert found.layer_forward_ms == pytest.approx(1.23)
            assert found.prefill_tokens_per_sec == pytest.approx(5000.0)
        finally:
            os.unlink(db_path)

    def test_save_update(self):
        """Test that saving with same key updates the record."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            # Save first version
            result1 = ProfileResult(
                model_name="test",
                dtype="float16",
                device_type="cuda:0",
                layer_forward_ms=1.0,
            )
            store.save(result1)

            # Save updated version
            result2 = ProfileResult(
                model_name="test",
                dtype="float16",
                device_type="cuda:0",
                layer_forward_ms=2.0,
            )
            store.save(result2)

            # Should have only one record, with updated value
            profiles = store.list_profiles()
            assert len(profiles) == 1
            assert profiles[0].layer_forward_ms == pytest.approx(2.0)
        finally:
            os.unlink(db_path)

    def test_list_profiles(self):
        """Test listing all profiles."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            # Save multiple profiles
            for i in range(3):
                result = ProfileResult(
                    model_name=f"model-{i}",
                    dtype="float16",
                    device_type="cuda:0",
                    layer_forward_ms=float(i),
                )
                store.save(result)

            profiles = store.list_profiles()
            assert len(profiles) == 3
        finally:
            os.unlink(db_path)

    def test_delete_profile(self):
        """Test deleting a profile."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            result = ProfileResult(
                model_name="to-delete",
                dtype="float16",
                device_type="cuda:0",
            )
            store.save(result)

            # Delete
            deleted = store.delete("to-delete")
            assert deleted

            # Verify gone
            profiles = store.list_profiles()
            assert len(profiles) == 0
        finally:
            os.unlink(db_path)

    def test_lookup_not_found(self):
        """Test looking up a non-existent profile."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            found = store.lookup(model_name="nonexistent")
            assert found is None
        finally:
            os.unlink(db_path)

    def test_partial_lookup(self):
        """Test looking up with partial key (only model_name)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ProfileStore(db_path)

            result = ProfileResult(
                model_name="test-model",
                dtype="float16",
                device_type="cuda:0",
                device_name="GPU-A",
                layer_forward_ms=1.5,
            )
            store.save(result)

            # Lookup by model_name only
            found = store.lookup(model_name="test-model")
            assert found is not None
            assert found.layer_forward_ms == pytest.approx(1.5)
        finally:
            os.unlink(db_path)


class TestProtobufIntegration:
    """Test protobuf ProfileResponse integration."""

    def test_profile_response_fields(self):
        """Test that ProfileResponse has all required fields."""
        from edgeshard._grpc import edgeshard_pb2

        response = edgeshard_pb2.ProfileResponse(
            success=True,
            message="OK",
            layer_forward_ms=1.5,
            kv_cache_per_token_mb=0.001,
            prefill_tokens_per_sec=5000.0,
            decode_tokens_per_sec=100.0,
            num_layers_profiled=24,
            total_model_memory_mb=1024.0,
            device_name="RTX 4090",
        )

        assert response.success
        assert response.layer_forward_ms == pytest.approx(1.5)
        assert response.num_layers_profiled == 24
        assert response.device_name == "RTX 4090"

    def test_profile_request_fields(self):
        """Test ProfileRequest fields."""
        from edgeshard._grpc import edgeshard_pb2

        request = edgeshard_pb2.ProfileRequest(
            model_name="Qwen/Qwen2.5-0.5B",
            model_revision="main",
            dtype="float16",
            device="cuda:0",
        )

        assert request.model_name == "Qwen/Qwen2.5-0.5B"
        assert request.device == "cuda:0"
