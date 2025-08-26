#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyPI Release Script for Chinese Herbal Medicine Sentiment Analysis Package

This script automates the process of building and releasing the package to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command and optionally check for errors."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result

def clean_build_dirs():
    """Clean previous build directories."""
    print("Cleaning previous build directories...")
    
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    
    for pattern in dirs_to_clean:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
        else:
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"Removed directory: {pattern}")

def check_requirements():
    """Check if required tools are installed."""
    print("Checking requirements...")
    
    required_tools = ['twine', 'wheel', 'setuptools']
    
    for tool in required_tools:
        result = run_command(f"python -c 'import {tool}'", check=False)
        if result.returncode != 0:
            print(f"Installing {tool}...")
            run_command(f"pip install {tool}")

def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Check if pytest is available
    result = run_command("python -c 'import pytest'", check=False)
    if result.returncode != 0:
        print("Installing pytest...")
        run_command("pip install pytest")
    
    # Run tests
    test_result = run_command("python -m pytest tests/ -v", check=False)
    
    if test_result.returncode != 0:
        print("WARNING: Some tests failed. Continue anyway? (y/N)")
        response = input().lower()
        if response != 'y':
            print("Aborting release due to test failures.")
            sys.exit(1)

def check_version():
    """Check and display the current version."""
    print("Checking version...")
    
    # Get version from __init__.py
    init_file = Path("chinese_herbal_sentiment/__init__.py")
    
    if not init_file.exists():
        print("Error: Cannot find chinese_herbal_sentiment/__init__.py")
        sys.exit(1)
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    import re
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    
    if not version_match:
        print("Error: Cannot find version in __init__.py")
        sys.exit(1)
    
    version = version_match.group(1)
    print(f"Package version: {version}")
    
    return version

def build_package():
    """Build the package."""
    print("Building package...")
    
    # Build source distribution
    run_command("python setup.py sdist")
    
    # Build wheel distribution
    run_command("python setup.py bdist_wheel")
    
    print("Package built successfully!")

def check_package():
    """Check the built package using twine."""
    print("Checking package with twine...")
    
    run_command("twine check dist/*")
    
    print("Package check passed!")

def upload_to_test_pypi():
    """Upload to TestPyPI for testing."""
    print("Uploading to TestPyPI...")
    
    print("Note: You'll need TestPyPI credentials.")
    print("Sign up at: https://test.pypi.org/account/register/")
    
    result = run_command("twine upload --repository testpypi dist/*", check=False)
    
    if result.returncode == 0:
        print("Successfully uploaded to TestPyPI!")
        print("Test installation with:")
        print("pip install --index-url https://test.pypi.org/simple/ chinese-herbal-sentiment")
    else:
        print("Failed to upload to TestPyPI. This might be expected if the version already exists.")
    
    return result.returncode == 0

def upload_to_pypi():
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    
    print("WARNING: This will upload to the official PyPI. Continue? (y/N)")
    response = input().lower()
    
    if response != 'y':
        print("Aborting PyPI upload.")
        return False
    
    print("Note: You'll need PyPI credentials.")
    print("Sign up at: https://pypi.org/account/register/")
    
    result = run_command("twine upload dist/*", check=False)
    
    if result.returncode == 0:
        print("Successfully uploaded to PyPI!")
        print("Install with: pip install chinese-herbal-sentiment")
        return True
    else:
        print("Failed to upload to PyPI.")
        return False

def main():
    """Main release process."""
    print("=" * 60)
    print("Chinese Herbal Medicine Sentiment Analysis - PyPI Release")
    print("=" * 60)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Step 1: Check requirements
        check_requirements()
        
        # Step 2: Check version
        version = check_version()
        
        # Step 3: Clean previous builds
        clean_build_dirs()
        
        # Step 4: Run tests
        run_tests()
        
        # Step 5: Build package
        build_package()
        
        # Step 6: Check package
        check_package()
        
        # Step 7: Ask user what to do
        print("\nPackage is ready for release!")
        print("Choose an option:")
        print("1. Upload to TestPyPI only")
        print("2. Upload to both TestPyPI and PyPI")
        print("3. Skip upload (build only)")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            upload_to_test_pypi()
        elif choice == "2":
            test_success = upload_to_test_pypi()
            if test_success:
                print("\nTestPyPI upload successful. Proceeding to PyPI...")
                upload_to_pypi()
            else:
                print("TestPyPI upload failed. Skipping PyPI upload.")
        elif choice == "3":
            print("Build completed. Skipping upload.")
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Release process completed!")
        print("=" * 60)
        
        # Show final information
        print(f"\nPackage: chinese-herbal-sentiment v{version}")
        print("Built files:")
        
        dist_dir = Path("dist")
        if dist_dir.exists():
            for file in dist_dir.iterdir():
                print(f"  - {file.name}")
        
        print("\nUseful links:")
        print("- GitHub: https://github.com/chenxingqiang/chinese-herbal-sentiment")
        print("- PyPI: https://pypi.org/project/chinese-herbal-sentiment/")
        print("- TestPyPI: https://test.pypi.org/project/chinese-herbal-sentiment/")
        print("- Dataset: https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment")
        
    except KeyboardInterrupt:
        print("\nRelease process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
