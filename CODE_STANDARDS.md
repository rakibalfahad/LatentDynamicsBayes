# Code Standards

This document outlines the coding standards for the LatentDynamicsBayes project.

## PEP 8 Compliance

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. The main points are:

### 1. Indentation
- Use 4 spaces per indentation level
- Continuation lines should align wrapped elements with a hanging indent

### 2. Line Length
- Limit all lines to a maximum of 79 characters for code
- Limit docstrings/comments to 72 characters

### 3. Imports
- Imports should be on separate lines
- Imports should be grouped in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Imports should be ordered alphabetically within each group

### 4. Whitespace
- Avoid extraneous whitespace
- No trailing whitespace
- No blank lines with whitespace
- Use blank lines to separate logical sections

### 5. Comments
- Comments should be complete sentences
- Use inline comments sparingly
- Update comments when code changes

### 6. Naming Conventions
- Function names should be lowercase, with words separated by underscores
- Variable names follow the same convention as function names
- Class names should use CamelCase (CapWords) convention
- Constants should be in ALL_CAPS with underscores

## Flake8 Configuration

We use Flake8 to enforce our code standards. Our configuration is:

```ini
[flake8]
max-line-length = 79
exclude = .git,__pycache__,build,dist,.venv
```

## How to Check Code

Run the following command to check code against our standards:

```bash
flake8 .
```

## Common Issues and How to Fix Them

### 1. Line Length (E501)
```python
# Bad
very_long_string = "This is an extremely long string that exceeds the 79 character limit specified by PEP 8."

# Good
very_long_string = (
    "This is an extremely long string that has been "
    "broken into multiple lines to adhere to PEP 8."
)
```

### 2. Whitespace in Blank Lines (W293)
```python
# Bad
def function():
    
    return result

# Good
def function():

    return result
```

### 3. Trailing Whitespace (W291)
```python
# Bad
def function():     
    return result

# Good
def function():
    return result
```

### 4. Unused Imports (F401)
```python
# Bad
import numpy as np
import pandas as pd  # Never used

# Good
import numpy as np
# Only import what you use
```

### 5. Function/Class Spacing (E302, E305)
```python
# Bad
def function1():
    return result
def function2():  # Missing two blank lines
    return result

# Good
def function1():
    return result


def function2():  # Two blank lines between functions
    return result
```

## Automatic Fixing

Some issues can be automatically fixed using tools like `autopep8`:

```bash
pip install autopep8
autopep8 --in-place --aggressive --aggressive *.py
```

Or using `black`:

```bash
pip install black
black .
```

## Pre-commit Hooks

Consider setting up pre-commit hooks to ensure code meets standards before committing:

```bash
pip install pre-commit
pre-commit install
```

Create a `.pre-commit-config.yaml` file:
```yaml
repos:
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
```

## Regular Code Reviews

Conduct regular code reviews to ensure adherence to standards. This improves code quality and knowledge sharing across the team.
