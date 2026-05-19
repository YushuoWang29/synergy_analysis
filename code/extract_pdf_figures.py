#!/usr/bin/env python3
"""
extract_pdf_figures.py
======================
Extract figures from an academic PDF, match them with their captions,
and generate a LaTeX document reorganising the important figures.

Usage:
    python extract_pdf_figures.py <path_to_pdf>

Output:
    ./pdf_extract/<pdf_name>/
        ├── extracted_figs/       # All extracted figures as PNG
        │   ├── fig_001.png
        │   ├── fig_002.png
        │   └── ...
        ├── extracted_text.md     # Full extracted text for reference
        ├── important_figures.json # Machine-selected important figures
        └── report.tex            # LaTeX document with selected figures
"""

import sys, os, json, re
from collections import defaultdict

import fitz  # PyMuPDF

# ========== CONFIG ==========
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pdf_extract')
DPI = 300          # Resolution for vector graphics
MAX_FIG_W = 500    # Max width in pixels (will be resized proportionally)
# ============================

# ---- Helper: Natural key for sorting ----
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# ---- Step 1: Extract text from PDF ----
def extract_text(pdf_path):
    """Extract full text from PDF, page by page."""
    doc = fitz.open(pdf_path)
    pages_text = {}
    for i, page in enumerate(doc):
        pages_text[i] = page.get_text("text")
    doc.close()
    return pages_text


# ---- Step 2: Extract all embedded images ----
def extract_images(pdf_path, output_dir):
    """Extract all embedded images from the PDF and save as PNG.
    Returns list of dicts: {page, index, path, width, height, xref}
    """
    img_dir = os.path.join(output_dir, 'extracted_figs')
    os.makedirs(img_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext   = base_image["ext"]     # png, jpeg, etc.
            img_w     = base_image["width"]
            img_h     = base_image["height"]

            # Skip very small images (likely icons, decorations)
            if img_w < 100 or img_h < 100:
                continue

            filename = f"fig_{page_num+1:03d}_{img_idx+1:02d}.{img_ext}"
            filepath = os.path.join(img_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_bytes)

            images.append({
                "page": page_num + 1,
                "index": img_idx,
                "path": filepath,
                "width": img_w,
                "height": img_h,
                "xref": xref,
            })

    doc.close()
    return images


# ---- Step 3: Extract full-page renderings for visual context ----
def render_full_pages(pdf_path, output_dir):
    """Render each page to a high-res image to capture figures embedded
    as vector graphics (important for LaTeX-generated figures)."""
    page_dir = os.path.join(output_dir, 'page_renders')
    os.makedirs(page_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        fp = os.path.join(page_dir, f"page_{i+1:03d}.png")
        pix.save(fp)
        paths.append(fp)
    doc.close()
    return paths


# ---- Step 4: Parse figure captions from text ----
def parse_figure_captions(pages_text):
    """Parse figure captions from extracted text.
    Uses patterns like 'Fig. 1', 'Figure 1', etc.
    """
    captions = []
    # Pattern: "Fig. 1 ..." or "Figure 1 ..." or "Fig. 1." etc.
    patterns = [
        r'(?:Fig\.|Figure|FIG\.|FIGURE)\s*(\d+(?:\.\d+)?)[\.\s]*([^\n]*(?:\n(?!\s*(?:Fig\.|Figure|Table|Section|Method|Result|Discussion|Conclusion|Reference))[^\n]*)*)',
    ]

    for page_num, text in pages_text.items():
        for pat in patterns:
            matches = re.finditer(pat, text, re.IGNORECASE)
            for m in matches:
                fig_num = m.group(1)
                caption = m.group(2).strip()
                # Clean up
                caption = re.sub(r'\s+', ' ', caption)
                if len(caption) > 10:  # Ignore very short matches
                    captions.append({
                        "page": page_num + 1,
                        "figure_number": fig_num,
                        "caption": caption,
                        "text_surrounding": _get_surrounding_text(text, m.start(), m.end()),
                    })

    # Deduplicate by figure number
    seen = set()
    unique_captions = []
    for cap in captions:
        key = cap["figure_number"]
        if key not in seen:
            seen.add(key)
            unique_captions.append(cap)
    return sorted(unique_captions, key=lambda x: natural_key(x["figure_number"]))


def _get_surrounding_text(full_text, start, end, window=500):
    """Get text around the match for context."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(full_text), end + window)
    surrounding = full_text[ctx_start:ctx_end]
    # Get the first reference to the figure in the surrounding text
    lines = surrounding.split('\n')
    result = []
    for line in lines:
        line = line.strip()
        if line:
            result.append(line)
    return ' '.join(result[-10:])   # last 10 non-empty lines


# ---- Step 5: Classify importance of figures ----
IMPORTANT_KEYWORDS = [
    'overview', 'schematic', 'architecture', 'pipeline', 'framework',
    'result', 'performance', 'comparison', 'experiment', 'analysis',
    'main result', 'key finding', 'proposed', 'our method', 'our approach',
    'example', 'illustration', 'demonstration', 'setup', 'system',
    'accuracy', 'error', 'confusion', 'curve', 'distribution',
    'improvement', 'state-of-the-art', 'benchmark', 'evaluation',
]

METHOD_KEYWORDS = ['method', 'approach', 'pipeline', 'architecture', 'overview', 'schematic']
RESULT_KEYWORDS = ['result', 'performance', 'accuracy', 'error', 'comparison', 'evaluation', 'experiment']


def classify_importance(caption):
    """Classify a figure as 'important', 'method', 'result', or 'supplementary'."""
    caption_lower = caption.lower()
    
    # Check for supplementary
    if re.search(r'\bS\d+\b|supplementary|supplement|appendix', caption_lower):
        return 'supplementary'
    
    # Check for method
    if any(kw in caption_lower for kw in METHOD_KEYWORDS):
        return 'method'
    
    # Check for results
    if any(kw in caption_lower for kw in RESULT_KEYWORDS):
        return 'result'
    
    # General importance
    importance_score = sum(1 for kw in IMPORTANT_KEYWORDS if kw in caption_lower)
    if importance_score >= 2:
        return 'important'
    elif importance_score == 1:
        return 'moderate'
    else:
        return 'general'


# ---- Step 6: Match extracted images with captions ----
def match_images_to_captions(images, captions, pages_text, output_dir):
    """Try to match extracted images to their captions.
    Also extract vector-graphics figures from page renders."""
    # Build page-to-images mapping
    images_by_page = defaultdict(list)
    for img in images:
        images_by_page[img["page"]].append(img)

    # For pages with captions but no embedded images, 
    # the figures might be vector graphics rendered in the page stream.
    # We'll use page renders for those.
    matched = []
    unmatched_images = []

    # Track which images we've used
    used_image_paths = set()

    for cap in captions:
        page = cap["page"]
        page_imgs = images_by_page.get(page, [])
        
        page_imgs_filtered = [img for img in page_imgs 
                              if img["path"] not in used_image_paths]
        
        # Take the first few images on the same page (likely the figure)
        # Figures are typically the largest images on the page
        if page_imgs_filtered:
            # Sort by size descending
            sorted_imgs = sorted(page_imgs_filtered, 
                                key=lambda x: x["width"] * x["height"], 
                                reverse=True)
            # Take the biggest
            selected = sorted_imgs[0]
            used_image_paths.add(selected["path"])
            
            matched.append({
                "figure_number": cap["figure_number"],
                "caption": cap["caption"],
                "page": page,
                "category": classify_importance(cap["caption"]),
                "image_path": selected["path"],
                "image_width": selected["width"],
                "image_height": selected["height"],
            })
        else:
            # No embedded image found on this page
            matched.append({
                "figure_number": cap["figure_number"],
                "caption": cap["caption"],
                "page": page,
                "category": classify_importance(cap["caption"]),
                "image_path": None,  # Will use page render
                "image_width": None,
                "image_height": None,
            })

    # Remaining unmatched images
    for img in images:
        if img["path"] not in used_image_paths:
            unmatched_images.append(img)

    return matched, unmatched_images


# ---- Step 7: Select important figures for the summary document ----
def select_important_figures(matched, max_figures=10):
    """Select the most important figures for a summary LaTeX document."""
    priority = {'result': 0, 'method': 1, 'important': 2, 'moderate': 3, 'general': 4, 'supplementary': 5}
    
    # Sort by category priority, then by figure number
    sorted_figs = sorted(matched, key=lambda x: (priority.get(x["category"], 99), 
                                                  natural_key(x["figure_number"])))
    
    # Take top figures, but ensure we cover at least one method and one result
    selected = []
    seen_categories = set()
    
    for fig in sorted_figs:
        if len(selected) >= max_figures:
            break
        selected.append(fig)
        seen_categories.add(fig["category"])
    
    # If we haven't got both a method and result, add more
    if 'method' not in seen_categories or 'result' not in seen_categories:
        for fig in sorted_figs:
            if len(selected) >= max_figures:
                break
            if fig not in selected:
                selected.append(fig)
                seen_categories.add(fig["category"])
    
    return selected


# ---- Step 8: Build LaTeX summary document ----
# ---- Helper: sanitize text for LaTeX ----
def sanitize_for_latex(text):
    """Remove non-printable characters and problematic Unicode that LaTeX can't handle.
    Also escape special LaTeX characters."""
    if not text:
        return ""
    
    # Remove control characters (0x00-0x1F except tab/newline/carriage return)
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if cp == 0x09 or cp == 0x0A or cp == 0x0D:  # tab, newline, return
            cleaned.append(ch)
        elif cp >= 0x20:                              # printable
            # Replace common Unicode punctuation with ASCII equivalents
            if cp in (0x2013, 0x2014):  # en dash, em dash
                cleaned.append('--')
            elif cp == 0x2018 or cp == 0x2019:  # single quotes
                cleaned.append("'")
            elif cp == 0x201C or cp == 0x201D:  # double quotes
                cleaned.append('"')
            elif cp in (0x0016, 0x0017, 0x0018, 0x0019):  # control chars
                cleaned.append(' ')
            elif cp == 0x2026:  # ellipsis
                cleaned.append('...')
            elif cp == 0x2032:  # prime
                cleaned.append("'")
            elif cp == 0x2033:  # double prime
                cleaned.append('"')
            elif cp == 0x00B0:  # degree
                cleaned.append('$^\\circ$')
            elif cp == 0x00B1:  # plus-minus
                cleaned.append('$\\pm$')
            elif cp == 0x00D7:  # multiplication
                cleaned.append('$\\times$')
            elif cp == 0x03B1:  # alpha
                cleaned.append('$\\alpha$')
            elif cp == 0x03B2:  # beta
                cleaned.append('$\\beta$')
            elif cp == 0x03BC:  # micro
                cleaned.append('$\\mu$')
            elif cp == 0x03C3:  # sigma
                cleaned.append('$\\sigma$')
            elif cp == 0x2264:  # <=
                cleaned.append('$\\le$')
            elif cp == 0x2265:  # >=
                cleaned.append('$\\ge$')
            elif cp == 0x2192:  # right arrow
                cleaned.append('$\\rightarrow$')
            elif cp == 0x2190:  # left arrow
                cleaned.append('$\\leftarrow$')
            elif cp == 0x00A0:  # non-breaking space
                cleaned.append(' ')
            elif cp > 0x7E and cp < 0x00A0:  # other control chars in this range
                cleaned.append(' ')
            elif cp > 0x02FF and cp < 0x1F00:  # skip combining marks and other blocks
                cleaned.append(' ')
            else:
                cleaned.append(ch)
        # else control characters < 0x20 (except tab/newline/return) -> strip
    
    text = ''.join(cleaned)
    
    # Escape LaTeX special chars
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')
    
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def build_summary_latex(selected_figures, pdf_name, page_render_paths, output_dir):
    """Generate a LaTeX document with the selected important figures."""
    
    # Escape LaTeX special characters in pdf_name
    safe_pdf_name = pdf_name.replace('_', r'\_').replace('%', r'\%')
    
    figs_dir = os.path.join(output_dir, 'extracted_figs')
    
    doc_lines = [
        r"\documentclass[11pt,a4paper,twocolumn]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{geometry}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\geometry{left=2cm,right=2cm,top=2.5cm,bottom=2.5cm}",
        "",
        r"\title{\textbf{Key Figures Extracted from:}\\",
        r"       \texttt{" + safe_pdf_name + r"}",
        r"}",
        r"\author{Automatically Extracted and Organised}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section*{Overview}",
        "This document presents the key figures automatically extracted from",
        r"\texttt{" + safe_pdf_name + r"}.",
        "Figures are categorised by type (method, result, or supplementary)",
        "and ordered by relevance.",
        "",
        r"\section*{Key Method Figures}",
    ]
    
    # Group by category
    method_figs = [f for f in selected_figures if f["category"] == 'method']
    result_figs = [f for f in selected_figures if f["category"] == 'result']
    other_figs  = [f for f in selected_figures if f["category"] not in ('method', 'result')]
    
    def write_figure_list(fig_list, doc_lines):
        for i, fig in enumerate(fig_list):
            caption_text = sanitize_for_latex(fig["caption"])
            # Truncate long captions
            if len(caption_text) > 500:
                caption_text = caption_text[:497] + '...'
            
            img_path = fig["image_path"]
            if img_path and os.path.exists(img_path):
                rel_path = os.path.relpath(img_path, output_dir).replace('\\', '/')
                doc_lines.append(f"  \\begin{{figure}}[H]")
                doc_lines.append(f"    \\centering")
                doc_lines.append(f"    \\includegraphics[width=\\linewidth]{{{rel_path}}}")
                doc_lines.append(f"    \\caption{{{caption_text}}}")
                doc_lines.append(f"    \\label{{fig:extracted_{i+1}}}")
                doc_lines.append(f"  \\end{{figure}}")
            else:
                # Use page render
                page = fig["page"]
                pr_paths = [p for p in page_render_paths 
                           if f"page_{page:03d}" in os.path.basename(p)]
                if pr_paths:
                    rel_path = os.path.relpath(pr_paths[0], output_dir).replace('\\', '/')
                    doc_lines.append(f"  \\begin{{figure}}[H]")
                    doc_lines.append(f"    \\centering")
                    doc_lines.append(f"    \\includegraphics[width=\\linewidth]{{{rel_path}}}")
                    doc_lines.append(f"    \\caption{{Page {page}: {caption_text}}}")
                    doc_lines.append(f"    \\label{{fig:extracted_{i+1}}}")
                    doc_lines.append(f"  \\end{{figure}}")
                else:
                    doc_lines.append(f"  \\begin{{figure}}[H]")
                    doc_lines.append(f"    \\centering")
                    doc_lines.append(f"    \\textbf{{[Figure from page {page} - not available as image]}}")
                    doc_lines.append(f"    \\caption{{{caption_text}}}")
                    doc_lines.append(f"    \\label{{fig:extracted_{i+1}}}")
                    doc_lines.append(f"  \\end{{figure}}")
    
    if method_figs:
        write_figure_list(method_figs, doc_lines)
    
    doc_lines.append("")
    doc_lines.append(r"\section*{Key Result Figures}")
    if result_figs:
        write_figure_list(result_figs, doc_lines)
    else:
        doc_lines.append("No result figures extracted.")
    
    doc_lines.append("")
    doc_lines.append(r"\section*{Other Important Figures}")
    if other_figs:
        write_figure_list(other_figs, doc_lines)
    else:
        doc_lines.append("No other figures extracted.")
    
    doc_lines.append("")
    doc_lines.append(r"\end{document}")
    
    return '\n'.join(doc_lines)


# ---- Main function ----
def main():
    if len(sys.argv) < 2:
        # Use default test
        pdf_path = input("Enter path to PDF: ").strip()
        if not pdf_path:
            print("Usage: python extract_pdf_figures.py <path_to_pdf>")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(OUTPUT_BASE, pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Extracting figures from: {os.path.basename(pdf_path)}")
    print(f"{'='*60}")
    
    # Step 1: Extract text
    print("\n>>> Extracting text from PDF...")
    pages_text = extract_text(pdf_path)
    print(f"    Extracted {len(pages_text)} pages of text.")
    
    # Save full text
    with open(os.path.join(output_dir, 'extracted_text.md'), 'w', encoding='utf-8') as f:
        for page_num, text in pages_text.items():
            f.write(f"\n\n=== PAGE {page_num+1} ===\n\n")
            f.write(text)
    print(f"    Saved to extracted_text.md")
    
    # Step 2: Extract images
    print("\n>>> Extracting embedded images...")
    images = extract_images(pdf_path, output_dir)
    print(f"    Found {len(images)} embedded images.")
    
    # Step 3: Render full pages
    print("\n>>> Rendering full pages (for vector figures)...")
    page_renders = render_full_pages(pdf_path, output_dir)
    print(f"    Rendered {len(page_renders)} pages.")
    
    # Step 4: Parse figure captions
    print("\n>>> Parsing figure captions...")
    captions = parse_figure_captions(pages_text)
    print(f"    Found {len(captions)} figure captions:")
    for cap in captions:
        print(f"        Fig. {cap['figure_number']} (p.{cap['page']}): {cap['caption'][:80]}...")
    
    # Step 5-6: Match images to captions
    print("\n>>> Matching images to captions...")
    matched, unmatched = match_images_to_captions(images, captions, pages_text, output_dir)
    print(f"    Matched {len(matched)} figures, {len(unmatched)} unmatched images.")
    
    # Step 7: Select important figures
    print("\n>>> Selecting important figures...")
    important = select_important_figures(matched, max_figures=12)
    
    # Categorise them
    categories = defaultdict(list)
    for fig in important:
        categories[fig["category"]].append(fig)
    
    print(f"    Selected {len(important)} important figures:")
    for cat, figs in categories.items():
        print(f"      [{cat}] {len(figs)} figures")
        for f in figs:
            print(f"        - Fig. {f['figure_number']}: {f['caption'][:60]}...")
    
    # Save important figures list
    important_data = []
    for fig in important:
        entry = {
            "figure_number": fig["figure_number"],
            "page": fig["page"],
            "category": fig["category"],
            "caption": fig["caption"],
            "image_path": fig["image_path"],
        }
        important_data.append(entry)
    
    with open(os.path.join(output_dir, 'important_figures.json'), 'w', encoding='utf-8') as f:
        json.dump(important_data, f, indent=2)
    print(f"    Saved to important_figures.json")
    
    # Step 8: Generate LaTeX
    print("\n>>> Generating LaTeX summary document...")
    latex_content = build_summary_latex(important, pdf_name, page_renders, output_dir)
    
    tex_path = os.path.join(output_dir, 'report.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"    Saved to report.tex")
    
    # Step 9: Optional compilation
    print("\n>>> Attempting LaTeX compilation...")
    latex_bin = None
    candidates = [
        r'D:\MiKTeX\miktex\bin\x64\pdflatex.exe',
        r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe',
        'pdflatex.exe',
    ]
    import shutil, subprocess
    for c in candidates:
        if shutil.which(c):
            latex_bin = c
            break
    
    if latex_bin:
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        try:
            for i in range(2):
                result = subprocess.run(
                    [latex_bin, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    print(f"    Pass {i+1} succeeded")
                else:
                    print(f"    Pass {i+1} had warnings (non-fatal)")
        except Exception as e:
            print(f"    Compilation error: {e}")
        finally:
            os.chdir(orig_dir)
        
        pdf_out = os.path.join(output_dir, 'report.pdf')
        if os.path.isfile(pdf_out):
            print(f"\n{'='*60}")
            print(f"  SUCCESS! Summary PDF generated:")
            print(f"  {pdf_out}")
            print(f"{'='*60}")
    else:
        print("    pdflatex not found. LaTeX source is ready for manual compilation.")
    
    print(f"\nDone! All output saved to:\n  {output_dir}")


if __name__ == '__main__':
    main()
