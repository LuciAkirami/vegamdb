"""Cross-platform test runner for cibuildwheel.

Copies tests to a temp directory so the source vegamdb/ directory
doesn't shadow the installed wheel package.

Usage: python scripts/run_tests.py <project_dir>
"""

import os
import shutil
import subprocess
import sys
import tempfile

project_dir = sys.argv[1]
tmp_dir = tempfile.mkdtemp()
test_dst = os.path.join(tmp_dir, "tests")

shutil.copytree(os.path.join(project_dir, "tests"), test_dst)
result = subprocess.run([sys.executable, "-m", "pytest", test_dst, "-v"])
shutil.rmtree(tmp_dir, ignore_errors=True)
sys.exit(result.returncode)
