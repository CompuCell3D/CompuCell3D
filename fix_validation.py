#!/usr/bin/env python3
"""
Script to fix validation in Spec_UI.py by replacing InputValidator with ErrorHandler
and removing complex validation logic
"""

import re

def fix_validation_file():
    file_path = "CompuCellJupyterInterfaceDevelopment/Feature Development/Spec_UI.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace InputValidator with ErrorHandler
    content = content.replace('InputValidator', 'ErrorHandler')
    
    # Remove validation setup calls - replace with just error display creation
    # Pattern: ErrorHandler.setup_input_validation(...) -> # Validation removed
    validation_pattern = r'ErrorHandler\.setup_input_validation\([^)]*\)\s*'
    content = re.sub(validation_pattern, '# Validation removed - handled by backend\n        ', content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove validation function definitions
    # Pattern: def validate_something(value): ... (multiple lines until next def or other structure)
    validate_func_pattern = r'def validate_[^(]*\([^)]*\):[^}]*?(?=\n        [a-zA-Z_]|\n    def |\n    #|\n    @|\n    class |\n        self\.|\n        [A-Z]|\n        return|\n        if|\n        for|\n        while|\n        try|\n        with|\Z)'
    content = re.sub(validate_func_pattern, '# Validation function removed\n        ', content, flags=re.MULTILINE | re.DOTALL)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed validation in Spec_UI.py")
    print("- Replaced InputValidator with ErrorHandler")
    print("- Removed validation setup calls")
    print("- Removed validation function definitions")

if __name__ == "__main__":
    fix_validation_file()
