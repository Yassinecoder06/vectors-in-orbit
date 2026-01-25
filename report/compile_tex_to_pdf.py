import subprocess
import sys
from pathlib import Path


def compile_tex_to_pdf(tex_path: str) -> None:
    """
    Compile a .tex file into PDF using a local LaTeX installation.

    Requires a LaTeX distribution installed and available on PATH:
    - Windows: MiKTeX or TeX Live
    - macOS: MacTeX
    - Linux: TeX Live
    """
    tex_file = Path(tex_path)
    if not tex_file.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_file}")

    # Use latexmk if available (recommended), otherwise fallback to pdflatex.
    latexmk_cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", str(tex_file)]
    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", str(tex_file)]

    try:
        print("Running latexmk...")
        subprocess.run(latexmk_cmd, check=True)
        print("PDF generated successfully.")
        return
    except FileNotFoundError:
        print("latexmk not found. Falling back to pdflatex...")
    except subprocess.CalledProcessError as e:
        print("latexmk failed. Falling back to pdflatex...")

    # Fallback: run pdflatex twice to resolve references
    subprocess.run(pdflatex_cmd, check=True)
    subprocess.run(pdflatex_cmd, check=True)
    print("PDF generated successfully with pdflatex.")


if __name__ == "__main__":
    # Default to pipeline_report.tex in current directory
    tex_path = "fa_cf_complete_implementation.tex"
    compile_tex_to_pdf(tex_path)
