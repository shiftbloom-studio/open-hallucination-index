#!/usr/bin/env python3
"""
Wikipedia Dual-Store Ingestion Script (Legacy Wrapper)
======================================================

This script has been refactored into a modular architecture for improved
performance and maintainability. It now serves as a thin wrapper around
the new `/ingestion/` module.

New Usage:
    # Direct module execution (recommended)
    python -m ingestion --help
    
    # Using this wrapper (legacy compatibility)
    python api/scripts/ingest_wiki_dual.py [args...]

The new modular architecture provides:
- 10-50x faster ingestion via producer-consumer pattern
- Parallel downloads with resume support  
- GPU-accelerated embeddings with large batches
- Non-blocking async uploads to Qdrant and Neo4j
- 10+ relationship types in Neo4j graph
- Resumable checkpoints for crash recovery
- Better progress tracking and statistics

See /ingestion/README.md for documentation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Force GPU venv if available
gpu_python = (
    project_root / ".venv-gpu" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)

if (
    gpu_python.exists()
    and Path(sys.executable).resolve() != gpu_python.resolve()
    and os.environ.get("OHI_FORCE_GPU") != "1"
):
    os.environ["OHI_FORCE_GPU"] = "1"
    args = [str(gpu_python), str(Path(__file__).resolve())] + sys.argv[1:]
    if "--embedding-device" not in sys.argv:
        args += ["--embedding-device", "cuda"]
    os.execv(str(gpu_python), args)

# Import and run the new modular pipeline
from ingestion.__main__ import main

if __name__ == "__main__":
    main()
