"""
chroma_compat.py — Safe ChromaDB client factory

Handles version differences and gives a clear error if chromadb-client
(HTTP-only) is installed instead of the full chromadb package.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def get_chroma_client(path: str):
    """
    Returns a ChromaDB PersistentClient, with helpful error messages
    for the common 'http-only client mode' failure.
    """
    try:
        import chromadb
    except ImportError:
        print("\n❌ chromadb not installed. Run: pip install 'chromadb>=0.5.0'")
        sys.exit(1)

    # Detect http-only client (chromadb-client package)
    _check_not_http_only(chromadb)

    # Warn if bad env vars are set
    _check_env_vars()

    try:
        client = chromadb.PersistentClient(path=path)
        return client
    except AttributeError:
        # Very old version (<0.3.x) used chromadb.Client(Settings(...))
        try:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=path,
                anonymized_telemetry=False,
            ))
            return client
        except Exception as e:
            _show_fix_instructions(str(e))
            sys.exit(1)
    except Exception as e:
        _show_fix_instructions(str(e))
        sys.exit(1)


def _check_not_http_only(chromadb_module):
    """Detect if chromadb-client (HTTP-only) is installed."""
    try:
        # Full chromadb has PersistentClient; http-only client does not support it
        # but raises the specific error message we've seen
        pass
    except Exception:
        pass

    # Check if it's the http-only package by looking at what's importable
    try:
        from chromadb.api.fastapi import FastAPI  # noqa
        # If this imports without error, we're in http-only mode
        print("\n❌ You have 'chromadb-client' (HTTP-only) installed instead of full 'chromadb'.")
        _show_fix_instructions("http-only client installed")
        sys.exit(1)
    except (ImportError, AttributeError):
        pass  # Good — full package doesn't expose this the same way


def _check_env_vars():
    """Warn about env vars that force HTTP-only mode."""
    bad = {
        'CHROMA_API_IMPL': os.environ.get('CHROMA_API_IMPL'),
        'CHROMA_SERVER_HOST': os.environ.get('CHROMA_SERVER_HOST'),
    }
    for var, val in bad.items():
        if val and 'fastapi' in str(val).lower():
            logger.warning(
                f"⚠️  Env var {var}={val} forces HTTP-only ChromaDB mode. "
                f"Unset it: `unset {var}`"
            )


def _show_fix_instructions(error: str):
    print(f"""
❌ ChromaDB Error: {error}

This usually means 'chromadb-client' (HTTP-only) is installed instead
of the full 'chromadb' package.

Fix (run in your terminal):
──────────────────────────────────────────────────────
# Option 1 — Auto fix script:
  python fix_chromadb.py

# Option 2 — Manual fix:
  pip uninstall chromadb chromadb-client -y
  pip install 'chromadb>=0.5.0,<2.0.0'

# Option 3 — If env var issue:
  unset CHROMA_API_IMPL
  unset CHROMA_SERVER_HOST
──────────────────────────────────────────────────────
""")
