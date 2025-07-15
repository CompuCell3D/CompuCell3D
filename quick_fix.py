#!/usr/bin/env python3
"""
Quick fix for InputValidator references
"""

def fix_file():
    file_path = "CompuCellJupyterInterfaceDevelopment/Feature Development/Spec_UI.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple replacements
    content = content.replace('InputValidator.create_error_display()', 'ErrorHandler.create_error_display()')
    content = content.replace('InputValidator.setup_input_validation(', '# ErrorHandler.setup_input_validation(')
    
    # Remove validation function blocks
    import re
    
    # Remove validation functions and their calls
    lines = content.split('\n')
    new_lines = []
    skip_until_next_section = False
    
    for line in lines:
        # Skip validation function definitions
        if 'def validate_' in line:
            skip_until_next_section = True
            new_lines.append('        # Validation function removed')
            continue
        
        # Skip validation setup calls
        if 'ErrorHandler.setup_input_validation(' in line or 'InputValidator.setup_input_validation(' in line:
            skip_until_next_section = True
            new_lines.append('        # Validation setup removed')
            continue
        
        # Stop skipping when we hit a new section
        if skip_until_next_section:
            if (line.strip().startswith('self.') or 
                line.strip().startswith('def ') or
                line.strip().startswith('class ') or
                line.strip().startswith('return ') or
                line.strip().startswith('# ') and 'Build UI' in line or
                line.strip() == '' and len(new_lines) > 0 and new_lines[-1].strip() != ''):
                skip_until_next_section = False
            else:
                continue
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed InputValidator references")

if __name__ == "__main__":
    fix_file()
