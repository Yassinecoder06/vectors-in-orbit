#!/usr/bin/env python3
"""Fix Unicode emoji characters in LaTeX file."""

import sys

def fix_unicode_in_tex(input_file, output_file):
    """Replace Unicode emojis with LaTeX-compatible symbols."""
    
    replacements = {
        '✅': r'$\checkmark$',
        '⚠️': r'\textbf{[!]}',
        '⚠': r'\textbf{[!]}',
        '🎉': r'\textbf{[SUCCESS]}',
        '📦': r'\textbf{[PKG]}',
        '⏭️': r'\textbf{[SKIP]}',
        '⏭': r'\textbf{[SKIP]}',
        '️': '',  # Remove variation selector
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed Unicode characters in {input_file}")
    print(f"✓ Output written to {output_file}")

if __name__ == '__main__':
    input_file = 'fa_cf_complete_implementation.tex'
    output_file = 'fa_cf_complete_implementation.tex'
    fix_unicode_in_tex(input_file, output_file)
