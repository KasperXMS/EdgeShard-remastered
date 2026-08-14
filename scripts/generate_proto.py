#!/usr/bin/env python3
"""Generate Python gRPC code from protobuf definitions.

Usage:
    python scripts/generate_proto.py

Requirements:
    pip install grpcio-tools
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Generate gRPC Python code."""
    proto_dir = Path("proto")
    output_dir = Path("src/edgeshard/_grpc")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print("No .proto files found in proto/")
        sys.exit(1)

    print(f"Found {len(proto_files)} proto file(s)")

    # Generate Python code
    for proto_file in proto_files:
        print(f"Generating {proto_file.name}...")

        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error generating {proto_file.name}:")
            print(result.stderr)
            sys.exit(1)

    # Fix import paths in generated _grpc.py files
    # The protoc generates "import xxx_pb2" but we need "import edgeshard._grpc.xxx_pb2"
    for grpc_file in output_dir.glob("*_pb2_grpc.py"):
        print(f"Fixing imports in {grpc_file.name}...")
        content = grpc_file.read_text()
        # Replace "import xxx_pb2 as" with "import edgeshard._grpc.xxx_pb2 as"
        import re
        content = re.sub(
            r'^import (\w+_pb2) as',
            r'import edgeshard._grpc.\1 as',
            content,
            flags=re.MULTILINE,
        )
        grpc_file.write_text(content)

    # Create __init__.py
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Generated gRPC code for EdgeShard."""\n')

    print(f"Generated gRPC code in {output_dir}")
    print("Don't forget to import in your code:")
    print("  from edgeshard._grpc import edgeshard_pb2")
    print("  from edgeshard._grpc import edgeshard_pb2_grpc")


if __name__ == "__main__":
    main()
