#!/usr/bin/env python
"""
Code Standardization Script for LatentDynamicsBayes project.

This script helps standardize the codebase by checking for common issues
and providing suggestions for improvements. It checks for:
- Missing docstrings
- Inconsistent import ordering
- Inconsistent naming conventions
- Missing type annotations
- Other style issues

Usage:
    python standardize_code.py [--fix] [--path PATH]

Options:
    --fix       Automatically fix issues where possible
    --path PATH Specify a specific path to check (default: entire project)
"""

import os
import re
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("standardize")

# File patterns to include
INCLUDE_PATTERNS = [r'.*\.py$']

# File patterns to exclude
EXCLUDE_PATTERNS = [r'__pycache__', r'\.git', r'\.vscode']

# Import order groups
IMPORT_GROUPS = [
    # Standard library
    r'^import (?:os|sys|math|re|time|datetime|json|logging|collections|random|argparse|pathlib|glob|shutil|io)',
    r'^from (?:os|sys|math|re|time|datetime|json|logging|collections|random|argparse|pathlib|glob|shutil|io)',
    
    # Third-party packages
    r'^import (?:numpy|torch|matplotlib|seaborn|pandas|scipy|sklearn)',
    r'^from (?:numpy|torch|matplotlib|seaborn|pandas|scipy|sklearn)',
    
    # Local imports
    r'^import (?:src|hdp_hmm|live)',
    r'^from (?:src|hdp_hmm|live)'
]

def should_process_file(file_path: str) -> bool:
    """
    Determine if a file should be processed based on include/exclude patterns.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if the file should be processed, False otherwise
    """
    # Check exclude patterns first
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, file_path):
            return False
    
    # Then check include patterns
    for pattern in INCLUDE_PATTERNS:
        if re.search(pattern, file_path):
            return True
    
    return False

def get_python_files(base_path: str) -> List[str]:
    """
    Get all Python files in the given path.
    
    Args:
        base_path (str): Base directory to search
        
    Returns:
        List[str]: List of Python file paths
    """
    python_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if should_process_file(file_path):
                python_files.append(file_path)
    
    return python_files

def check_docstrings(content: str, file_path: str) -> List[str]:
    """
    Check for missing or incomplete docstrings.
    
    Args:
        content (str): File content
        file_path (str): Path to the file
        
    Returns:
        List[str]: List of issues found
    """
    issues = []
    
    # Check for module docstring
    if not re.search(r'^""".*?"""', content, re.DOTALL):
        issues.append(f"{file_path}: Missing module docstring")
    
    # Check for class docstrings
    class_matches = re.finditer(r'class\s+(\w+)', content)
    for match in class_matches:
        class_name = match.group(1)
        class_pos = match.end()
        
        # Look for docstring after class definition
        if not re.search(r'^\s+"""', content[class_pos:class_pos + 200].splitlines()[0]):
            issues.append(f"{file_path}: Missing docstring for class {class_name}")
    
    # Check for function docstrings
    func_matches = re.finditer(r'def\s+(\w+)\s*\(', content)
    for match in func_matches:
        func_name = match.group(1)
        func_pos = match.end()
        
        # Skip if it's a special method
        if func_name.startswith('__') and func_name.endswith('__'):
            continue
            
        # Look for docstring after function definition
        # Skip to next line to handle multiline function definitions
        next_line_pos = content.find('\n', func_pos) + 1
        if next_line_pos > 0:
            if not re.search(r'^\s+"""', content[next_line_pos:next_line_pos + 200].splitlines()[0]):
                issues.append(f"{file_path}: Missing docstring for function {func_name}")
    
    return issues

def check_import_order(content: str, file_path: str) -> List[str]:
    """
    Check for inconsistent import ordering.
    
    Args:
        content (str): File content
        file_path (str): Path to the file
        
    Returns:
        List[str]: List of issues found
    """
    issues = []
    
    # Extract import section
    import_section_match = re.search(r'((?:import|from).*?)\n\n', content, re.DOTALL)
    if not import_section_match:
        return issues
        
    import_section = import_section_match.group(1)
    import_lines = [line.strip() for line in import_section.splitlines() if line.strip()]
    
    # Check group ordering
    current_group = -1
    for line in import_lines:
        for group_idx, pattern in enumerate(IMPORT_GROUPS):
            if re.match(pattern, line):
                if group_idx < current_group:
                    issues.append(f"{file_path}: Import ordering issue: {line} should come before group {current_group}")
                current_group = max(current_group, group_idx)
                break
    
    return issues

def check_naming_conventions(content: str, file_path: str) -> List[str]:
    """
    Check for inconsistent naming conventions.
    
    Args:
        content (str): File content
        file_path (str): Path to the file
        
    Returns:
        List[str]: List of issues found
    """
    issues = []
    
    # Check class names (should be CamelCase)
    class_matches = re.finditer(r'class\s+(\w+)', content)
    for match in class_matches:
        class_name = match.group(1)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
            issues.append(f"{file_path}: Class name {class_name} should be CamelCase")
    
    # Check function and variable names (should be snake_case)
    func_matches = re.finditer(r'def\s+(\w+)\s*\(', content)
    for match in func_matches:
        func_name = match.group(1)
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            continue
        if not re.match(r'^[a-z][a-z0-9_]*$', func_name):
            issues.append(f"{file_path}: Function name {func_name} should be snake_case")
    
    # Check constants (should be UPPER_CASE)
    # Look for variables defined at module level that look like constants
    const_matches = re.finditer(r'^\s*([A-Z][A-Z0-9_]*)\s*=', content, re.MULTILINE)
    for match in const_matches:
        const_name = match.group(1)
        if not re.match(r'^[A-Z][A-Z0-9_]*$', const_name):
            issues.append(f"{file_path}: Constant name {const_name} should be UPPER_CASE")
    
    return issues

def check_type_annotations(content: str, file_path: str) -> List[str]:
    """
    Check for missing type annotations.
    
    Args:
        content (str): File content
        file_path (str): Path to the file
        
    Returns:
        List[str]: List of issues found
    """
    issues = []
    
    # Check if typing is imported
    has_typing_import = re.search(r'(?:from typing import|import typing)', content) is not None
    
    # Check function definitions for type annotations
    func_matches = re.finditer(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->.*?)?:', content, re.DOTALL)
    for match in func_matches:
        func_name = match.group(1)
        params = match.group(2)
        
        # Skip if it's a special method
        if func_name.startswith('__') and func_name.endswith('__'):
            continue
            
        # Check if any parameter lacks type annotation
        has_params = params.strip() != ''
        if has_params:
            # Split parameters and remove default values for checking
            param_list = [p.strip().split('=')[0].strip() for p in params.split(',')]
            
            for param in param_list:
                if param == 'self' or param == 'cls':
                    continue
                if ':' not in param:
                    issues.append(f"{file_path}: Missing type annotation for parameter {param} in function {func_name}")
        
        # Check if return type is annotated
        has_return_type = re.search(r'\)\s*->', match.group(0)) is not None
        if not has_return_type:
            issues.append(f"{file_path}: Missing return type annotation for function {func_name}")
    
    # If we found missing annotations but no typing import, suggest adding it
    if issues and not has_typing_import:
        issues.append(f"{file_path}: Missing import from typing module")
    
    return issues

def analyze_file(file_path: str) -> List[str]:
    """
    Analyze a file for style issues.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        List[str]: List of issues found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        issues = []
        
        # Check docstrings
        issues.extend(check_docstrings(content, file_path))
        
        # Check import order
        issues.extend(check_import_order(content, file_path))
        
        # Check naming conventions
        issues.extend(check_naming_conventions(content, file_path))
        
        # Check type annotations
        issues.extend(check_type_annotations(content, file_path))
        
        return issues
    except Exception as e:
        return [f"Error analyzing {file_path}: {str(e)}"]

def generate_module_docstring(file_path: str) -> str:
    """
    Generate a module docstring based on the file name.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Generated docstring
    """
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    
    # Convert snake_case to Title Case
    title = ' '.join(word.capitalize() for word in module_name.split('_'))
    
    docstring = f'"""\n{title} Module for HDP-HMM.\n\nThis module provides...\n"""\n\n'
    return docstring

def fix_file(file_path: str, issues: List[str]) -> bool:
    """
    Attempt to fix issues in a file.
    
    Args:
        file_path (str): Path to the file
        issues (List[str]): List of issues to fix
        
    Returns:
        bool: True if any fixes were applied, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified = False
        
        # Add module docstring if missing
        if any(issue.endswith('Missing module docstring') for issue in issues):
            docstring = generate_module_docstring(file_path)
            if not content.startswith('"""'):
                content = docstring + content
                modified = True
        
        # Add typing import if missing
        if any('Missing import from typing module' in issue for issue in issues):
            if 'from typing import' not in content:
                if 'import ' in content:
                    # Add after other imports
                    content = re.sub(
                        r'((?:import|from).*?\n\n)',
                        r'\1from typing import List, Dict, Optional, Tuple, Any\n\n',
                        content,
                        count=1,
                        flags=re.DOTALL
                    )
                else:
                    # Add at the beginning after docstring
                    docstring_match = re.search(r'^""".*?"""\n', content, re.DOTALL)
                    if docstring_match:
                        pos = docstring_match.end()
                        content = content[:pos] + '\nfrom typing import List, Dict, Optional, Tuple, Any\n' + content[pos:]
                    else:
                        content = 'from typing import List, Dict, Optional, Tuple, Any\n\n' + content
                modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize code in the project")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues where possible")
    parser.add_argument("--path", default=".", help="Specify a specific path to check")
    
    args = parser.parse_args()
    
    base_path = os.path.abspath(args.path)
    logger.info(f"Checking files in {base_path}")
    
    files = get_python_files(base_path)
    logger.info(f"Found {len(files)} Python files")
    
    all_issues = []
    fixed_files = []
    
    for file_path in files:
        issues = analyze_file(file_path)
        
        if issues:
            all_issues.extend(issues)
            
            if args.fix:
                if fix_file(file_path, issues):
                    fixed_files.append(file_path)
    
    # Report results
    if all_issues:
        logger.info(f"Found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.info(f"  - {issue}")
    else:
        logger.info("No issues found!")
    
    if args.fix and fixed_files:
        logger.info(f"Applied fixes to {len(fixed_files)} files:")
        for file in fixed_files:
            logger.info(f"  - {file}")
    
    if all_issues:
        logger.info("Some issues may require manual fixes. Check the output for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
