"""Shard daemon — standalone shard process for distributed inference.

A ShardDaemon:
1. Loads a model subset (layer range)
2. Starts a gRPC server for data-plane tensor transfers AND inference RPCs
3. Optionally registers with a Master
4. Waits for inference commands from remote clients

Usage:
    edgeshard shard start \\
        Qwen/Qwen2.5-7B-Instruct \\
        --shard-id shard-0 \\
        --layers 0:16 \\
        --port 50100 \\
        --first --last
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from edgeshard.common.config import WorkerConfig
from edgeshard.common.logging import get_logger, setup_logging
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.transport.grpc_transport import ShardServer

logger = get_logger(__name__)


class ShardDaemon:
    """Standalone shard process."""

    def __init__(
        self,
        shard_id: str,
        model_path: str,
        layer_start: int,
        layer_end: int,
        dtype: str = "float16",
        data_plane_host: str = "0.0.0.0",
        data_plane_port: int = 50100,
        is_first_shard: bool = False,
        is_last_shard: bool = False,
    ) -> None:
        self._shard_id = shard_id
        self._model_path = model_path
        self._layer_start = layer_start
        self._layer_end = layer_end
        self._dtype = dtype
        self._data_plane_host = data_plane_host
        self._data_plane_port = data_plane_port
        self._is_first_shard = is_first_shard
        self._is_last_shard = is_last_shard

        self._adapter: Qwen2Adapter | None = None
        self._shard: ModelShard | None = None
        self._server: ShardServer | None = None

    async def start(self) -> None:
        """Start the shard daemon."""
        logger.info(f"Starting shard {self._shard_id}")
        logger.info(f"  Model: {self._model_path}")
        logger.info(f"  Layers: [{self._layer_start}, {self._layer_end})")
        logger.info(f"  First shard: {self._is_first_shard}")
        logger.info(f"  Last shard: {self._is_last_shard}")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self._dtype, torch.float16)

        logger.info(f"Loading model on {device} with dtype {dtype}")

        # Auto-download model from HuggingFace if not found locally
        self._model_path = self._ensure_model_available(self._model_path)

        self._adapter = Qwen2Adapter()
        self._adapter.load(
            model_path=self._model_path,
            layer_start=self._layer_start,
            layer_end=self._layer_end,
            dtype=dtype,
            device=device,
        )

        # Create shard
        self._shard = ModelShard(
            shard_id=self._shard_id,
            adapter=self._adapter,
            is_first_shard=self._is_first_shard,
            is_last_shard=self._is_last_shard,
        )

        # Start data-plane server WITH inference RPCs enabled
        self._server = ShardServer(
            shard_id=self._shard_id,
            host=self._data_plane_host,
            port=self._data_plane_port,
            inference_shard=self._shard,
        )
        await self._server.start()

        logger.info(f"Shard {self._shard_id} started successfully")

    async def stop(self) -> None:
        """Stop the shard daemon."""
        logger.info(f"Stopping shard {self._shard_id}")

        if self._server:
            await self._server.stop()

        if self._adapter:
            self._adapter.unload()

        logger.info(f"Shard {self._shard_id} stopped")

    def get_shard(self) -> ModelShard | None:
        """Get the ModelShard instance (for testing)."""
        return self._shard

    def _ensure_model_available(self, model_path: str) -> str:
        """Ensure model is available locally, downloading from HuggingFace if needed.

        Args:
            model_path: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
                or local path.

        Returns:
            Local path to the model (downloaded if necessary).
        """
        # Check if it's a local path that exists
        local_path = Path(model_path)
        if local_path.exists() and (local_path / "config.json").exists():
            return model_path

        # Check under models/ directory
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        models_dir = Path("models") / model_name
        if models_dir.exists() and (models_dir / "config.json").exists():
            return str(models_dir)

        # Check HF cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        # HF cache format: models--<org>--<model>
        cache_name = f"models--{model_path.replace('/', '--')}"
        cache_dir = hf_cache / cache_name
        if cache_dir.exists():
            # Find the snapshot directory
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if (snapshot / "config.json").exists():
                        logger.info(f"Found model in HF cache: {snapshot}")
                        return str(snapshot)

        # Model not found locally — try to download from HuggingFace
        if "/" in model_path:
            logger.info(f"Model not found locally. Downloading {model_path} from HuggingFace...")
            try:
                from huggingface_hub import snapshot_download

                local_dir = Path("models") / model_name
                local_dir.mkdir(parents=True, exist_ok=True)

                downloaded_path = snapshot_download(
                    repo_id=model_path,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Model downloaded to: {downloaded_path}")
                return downloaded_path

            except ImportError:
                logger.error(
                    "huggingface_hub not installed. Install with: pip install huggingface_hub"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to download model from HuggingFace: {e}")
                raise

        # Not a HF model ID and not found locally
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Provide a valid local path or HuggingFace model ID."
        )


def main() -> None:
    """CLI entry point for shard daemon.

    Usage:
        python -m edgeshard.worker.shard_daemon \\
            Qwen/Qwen2.5-7B-Instruct \\
            --shard-id shard-0 \\
            --layers 0:16 \\
            --port 50100 \\
            --first
    """
    import argparse

    parser = argparse.ArgumentParser(description="EdgeShard Shard Daemon")
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("--shard-id", required=True, help="Unique shard ID")
    parser.add_argument("--layers", required=True, help="Layer range, e.g. 0:16")
    parser.add_argument("--dtype", default="float16", help="Model dtype")
    parser.add_argument("--host", default="0.0.0.0", help="Data plane host")
    parser.add_argument("--port", type=int, default=50100, help="Data plane port")
    parser.add_argument("--first", action="store_true", help="This is the first shard (has embedding)")
    parser.add_argument("--last", action="store_true", help="This is the last shard (has LM head)")

    args = parser.parse_args()

    # Parse layer range
    layer_parts = args.layers.split(":")
    layer_start = int(layer_parts[0])
    layer_end = int(layer_parts[1])

    daemon = ShardDaemon(
        shard_id=args.shard_id,
        model_path=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        dtype=args.dtype,
        data_plane_host=args.host,
        data_plane_port=args.port,
        is_first_shard=args.first,
        is_last_shard=args.last,
    )

    async def run() -> None:
        await daemon.start()
        logger.info(f"Shard daemon running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await daemon.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
