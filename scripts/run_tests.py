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
from pathlib import Path

def run():
    # 1. Normalize the project path (handles Windows Short Names)
    project_dir = Path(sys.argv[1]).resolve()
    
    # 2. Create a clean temp directory
    tmp_dir = Path(tempfile.mkdtemp()).resolve()
    test_dst = tmp_dir / "tests"

    try:
        # 3. Copy tests to the "Clean Island"
        shutil.copytree(project_dir / "tests", test_dst)

        # 4. CRITICAL: Change directory into the temp folder
        # This makes the temp folder the "root" for pytest
        os.chdir(tmp_dir)

        # 5. Run pytest on the LOCAL 'tests' directory
        # We use 'tests' (relative) instead of an absolute path to avoid path-string conflicts
        print(f"Running tests in: {os.getcwd()}")
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"])
        
        sys.exit(result.returncode)

    finally:
        # 6. Cleanup even if tests fail
        os.chdir(project_dir) # Move out so we can delete the folder
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    run()