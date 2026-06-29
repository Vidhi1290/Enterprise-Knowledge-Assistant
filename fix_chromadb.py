#!/usr/bin/env python3
"""
fix_chromadb.py — Run this once to fix the ChromaDB 'http-only client mode' error.

The error happens when:
  1. 'chromadb-client' (HTTP-only package) is installed instead of full 'chromadb'
  2. Old chromadb version (<0.4.x) that has different API
  3. CHROMA_API_IMPL env var is set to an HTTP client

Run:
  python fix_chromadb.py
"""

import subprocess
import sys
import os


def run(cmd):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.stderr.strip() and result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
    return result.returncode == 0


print("\n🔧 Fixing ChromaDB installation...\n")

# Step 1: Check what's installed
print("Step 1: Checking installed packages...")
result = subprocess.run("pip list", shell=True, capture_output=True, text=True)
chroma_pkgs = [l for l in result.stdout.split('\n') if 'chroma' in l.lower()]
print(f"  Found: {chroma_pkgs}")

# Step 2: Uninstall all chroma variants
print("\nStep 2: Removing all chromadb variants...")
run("pip uninstall chromadb chromadb-client chroma -y 2>/dev/null || true")

# Step 3: Install the correct full package
print("\nStep 3: Installing full chromadb...")
success = run(f"{sys.executable} -m pip install 'chromadb>=0.5.0,<2.0.0' --force-reinstall -q")

if not success:
    print("\n❌ pip install failed. Try manually:")
    print("   pip install 'chromadb>=0.5.0,<2.0.0' --force-reinstall")
    sys.exit(1)

# Step 4: Check for env vars that force HTTP mode
print("\nStep 4: Checking environment variables...")
bad_vars = ['CHROMA_API_IMPL', 'CHROMA_SERVER_HOST', 'CHROMA_SERVER_HTTP_PORT']
found_bad = False
for var in bad_vars:
    val = os.environ.get(var)
    if val:
        print(f"  ⚠️  Found {var}={val} — this forces HTTP-only mode!")
        print(f"     Unset it: unset {var}")
        found_bad = True
if not found_bad:
    print("  ✓ No problematic env vars found")

# Step 5: Verify fix
print("\nStep 5: Verifying fix...")
verify_script = """
import chromadb
import tempfile, os
tmpdir = tempfile.mkdtemp()
client = chromadb.PersistentClient(path=tmpdir)
col = client.get_or_create_collection('test', metadata={'hnsw:space': 'cosine'})
col.add(ids=['t1'], documents=['test doc'], embeddings=[[0.1, 0.2, 0.3]])
count = col.count()
client.delete_collection('test')
import shutil; shutil.rmtree(tmpdir)
print(f'ChromaDB {chromadb.__version__} working correctly (count={count})')
"""

result = subprocess.run([sys.executable, "-c", verify_script], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✅ {result.stdout.strip()}")
    print("\n✅ Fix successful! Now run:")
    print("   streamlit run streamlit_app.py")
else:
    print(f"  ❌ Still failing: {result.stderr.strip()}")
    print("\n  Manual fix:")
    print("  1. pip uninstall chromadb chromadb-client -y")
    print("  2. pip install chromadb==0.5.23")
    print("  3. Check: unset CHROMA_API_IMPL")
    sys.exit(1)
