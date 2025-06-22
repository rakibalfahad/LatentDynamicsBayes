#!/usr/bin/env python
"""
Script to automatically fix common PEP 8 issues in the codebase.
This script uses autopep8 to automatically fix formatting issues.
"""

import os
import subprocess
import argparse

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='Fix PEP 8 issues in Python files')
    parser.add_argument('--aggressive', '-a', action='count', default=0,
                        help='Enable aggressive fixes with autopep8 (use -aa for very aggressive)')
    parser.add_argument('--check', '-c', action='store_true',
                        help='Only check for issues without fixing them')
    parser.add_argument('--path', '-p', default='.',
                        help='Path to the directory containing Python files to fix')
    return parser

def install_dependencies():
    """Install required dependencies if they don't exist."""
    try:
        import autopep8
    except ImportError:
        print("Installing autopep8...")
        subprocess.check_call(["pip", "install", "autopep8"])

def find_python_files(directory):
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .venv directories
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        if '.venv' in dirs:
            dirs.remove('.venv')
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def fix_pep8_issues(files, aggressive_level=0, check_only=False):
    """Fix PEP 8 issues in the given files using autopep8."""
    import autopep8
    
    fixed_files = 0
    total_issues = 0
    
    for file_path in files:
        print(f"Processing {file_path}...")
        
        # Read the file content
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        # Generate autopep8 command options
        options = autopep8.parse_args(['--max-line-length', '79'])
        options.aggressive = aggressive_level
        options.in_place = not check_only
        options.verbose = 0
        
        # Fix the file
        fixed_content = autopep8.fix_code(original_content, options)
        
        # Count differences as issues
        issues = sum(1 for a, b in zip(original_content.splitlines(), 
                                   fixed_content.splitlines()) if a != b)
        total_issues += issues
        
        if issues > 0:
            print(f"  Found {issues} issues")
            if not check_only:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                fixed_files += 1
                print(f"  Fixed {issues} issues")
        else:
            print("  No issues found")
    
    return fixed_files, total_issues

def main():
    """Main function to fix PEP 8 issues in the codebase."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Install dependencies
    install_dependencies()
    
    # Find Python files
    python_files = find_python_files(args.path)
    print(f"Found {len(python_files)} Python files")
    
    # Fix PEP 8 issues
    fixed_files, total_issues = fix_pep8_issues(
        python_files, 
        aggressive_level=args.aggressive,
        check_only=args.check
    )
    
    # Print summary
    if args.check:
        print(f"\nFound {total_issues} issues in {fixed_files} files")
    else:
        print(f"\nFixed {total_issues} issues in {fixed_files} files")

if __name__ == "__main__":
    main()
