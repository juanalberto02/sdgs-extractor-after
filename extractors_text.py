# extractors_text.py — simple, plug-and-play
import re
import fitz  # PyMuPDF

# =====================
# ====== TITLE =========
# =====================
def extract_title_full_span(pdf_path, max_pages=1, debug=False):
    doc = fitz.open(pdf_path)
    spans = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    size = span["size"]
                    y0 = span["bbox"][1]
                    y1 = span["bbox"][3]
                    if len(text) > 0:
                        spans.append({
                            "text": text,
                            "size": size,
                            "y0": y0,
                            "y1": y1
                        })

    if not spans:
        return "Title not found"

    # Ambil semua ukuran font, urut dari besar ke kecil
    font_sizes = sorted(set(span["size"] for span in spans), reverse=True)

    for size in font_sizes:
        candidate_spans = [s for s in spans if abs(s["size"] - size) < 0.5]
        candidate_spans.sort(key=lambda s: s["y0"])

        # Lewati kandidat kalau semua baris terlalu pendek
        if all(len(s["text"].split()) <= 2 for s in candidate_spans):
            continue

        title_lines = []
        prev_y1 = None
        for s in candidate_spans:
            word_count = len(s["text"].split())

            # Lewati baris pembuka yang terlalu atas dan hanya sedikit kata
            if prev_y1 is None:
                if word_count < 3 or s["y0"] < 100:
                    continue
                title_lines.append(s)
                prev_y1 = s["y1"]
                continue

            vertical_gap = s["y0"] - prev_y1
            if vertical_gap < 20:
                title_lines.append(s)
                prev_y1 = s["y1"]
            else:
                break

        # Gabungkan dan cek jumlah kata
        title = " ".join(s["text"] for s in title_lines).strip()
        if len(title.split()) >= 5:
            if debug:
                print(f"[DEBUG] Selected font size: {size}")
                for s in title_lines:
                    print(f"[DEBUG] '{s['text']}' | size={s['size']} y0={s['y0']:.2f}")
                print(f"[DEBUG] Final Title: {title}")
            return title

    return "Title not found"


# =====================
# ===== ABSTRACT ======
# =====================

# ---------- UTILITIES ----------
def _normalize_line(line):
    return re.sub(r"[^a-z]", "", (line or "").lower())

def _matches_end_marker(line, end_keywords):
    line_lower = (line or "").strip().lower()
    for keyword in end_keywords:
        kw_lower = keyword.lower()
        if re.match(r"^(?:\d+\.\s*)?introduction\s*[:]?$", (line or "").strip(), re.IGNORECASE):
            return True
        if line_lower.startswith(kw_lower):
            return True
        if line_lower == kw_lower:
            return True
        if kw_lower in line_lower and len((line or '').strip().split()) <= 8:
            return True
        if re.search(rf"\b\d*\.?\s*{re.escape(kw_lower)}\b", line_lower):
            return True
    return False

# ---------- TEXT EXTRACTION ----------
def _extract_text_from_first_two_pages(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(min(2, len(doc))):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def _extract_text_from_first_two_pages_sorted_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(2, len(doc))):
        page = doc[i]
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by y0, then x0
        for block in blocks:
            content = block[4]
            if content.strip():
                text += content.strip() + "\n"
    return text

# ---------- ABSTRACT EXTRACTOR ----------
def _extract_abstract_from_text(text):
    lines = text.splitlines()
    start_index = None
    abstract_lines = []

    start_keywords = ["abstract", "summary"]
    end_keywords = [
        "keywords", "keyword", "1. introduction", "1. Introduction",
        "research in context", "acknowledgements", "references",
        "Funding Bill & Melinda Gates Foundation", "Competing interests:", "Citation"
    ]

    # --- Cari baris mulai abstract ---
    for i, line in enumerate(lines):
        for kw in start_keywords:
            if _normalize_line(kw) == _normalize_line(line):
                start_index = i + 1
                break
            elif _normalize_line(kw) in _normalize_line(line):
                match = re.search(rf"{kw}[\.:]?\s+(.*)", line, re.IGNORECASE)
                if match and match.group(1).strip():
                    abstract_lines.append(match.group(1).strip())
                start_index = i + 1
                break
        if start_index is not None:
            break

    # --- Fallback jika tidak ditemukan ---
    if start_index is None and not abstract_lines:
        for i, line in enumerate(lines):
            if re.search(r"(introduction|background|methods|results|conclusion)\s*[:\-]", (line or "").lower()):
                start_index = i
                break
        if start_index is None:
            return "Abstract or Summary not found"

    # --- Cari batas akhir abstract ---
    intro_indexes = []
    end_index = None

    for j in range(start_index, len(lines)):
        line = lines[j]
        if re.search(r"\bintroduction\b", (line or "").strip(), re.IGNORECASE):
            intro_indexes.append(j)
        if _matches_end_marker(line, end_keywords):
            end_index = j
            break

    # --- Jika hanya ada satu "Introduction", pakai itu sebagai akhir ---
    if end_index is None:
        if len(intro_indexes) == 1:
            end_index = intro_indexes[0]
        elif len(intro_indexes) >= 2:
            end_index = intro_indexes[1]

    # --- Ambil isi abstract ---
    end_limit = end_index if end_index is not None else len(lines)
    for j in range(start_index, end_limit):
        abstract_lines.append(lines[j])

    abstract_raw = " ".join(abstract_lines)

    # Potong inline '1. Introduction' bila menempel
    inline_intro = re.search(r"\b1\.\s*Introduction\b", abstract_raw, re.IGNORECASE)
    if inline_intro:
        abstract_raw = abstract_raw[:inline_intro.start()]

    abstract = re.sub(r"-\n", "", abstract_raw)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract if abstract else "Abstract not found"

# ---------- SMART CONTROLLER ----------
def extract_abstract_smart(pdf_path):
    try:
        text_blocks = _extract_text_from_first_two_pages_sorted_blocks(pdf_path)
        abstract_blocks = _extract_abstract_from_text(text_blocks)
        if "not found" not in abstract_blocks.lower() and len(abstract_blocks.split()) > 30:
            return abstract_blocks
    except Exception:
        pass

    text_simple = _extract_text_from_first_two_pages(pdf_path)
    abstract_simple = _extract_abstract_from_text(text_simple)
    if "not found" not in abstract_simple.lower() and len(abstract_simple.split()) > 30:
        return abstract_simple

    return "Abstract not found"


# =====================
# ===== KEYWORDS ======
# =====================

def extract_text_for_keywords(pdf_path, max_pages=3):
    """Raw text (hal 1..max_pages) untuk dipakai fungsi keywords downstream."""
    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text += page.get_text("text") + "\n"
    return text

def extract_keywords_from_text(text):
    """Ekstrak keywords dari raw text (sederhana & cepat)."""
    # Cari posisi awal "Keywords" atau "Index Terms"
    match = re.search(r"(?i)\b(?:Key\s*words|Keywords|Index\s*Terms)[\s—:-]*(.*)", text)
    if not match:
        return ""

    keywords_start = match.start()
    text_after_keywords = text[keywords_start:]

    # --- Cari batas akhir ---
    match_paren = re.search(r"\(", text_after_keywords)
    match_heading = re.search(
        r"\n(?:[IVXLCDM]+\.\s+|INTRODUCTION|METHODS|RESULTS|CONCLUSION|REFERENCES|DISCUSSION|ACKNOWLEDGMENT|MATERIALS AND METHODS|ARTICLE HISTORY|CONTACT|Introduction|Publisher|Highlight|Paper type|1. Introduction)\b",
        text_after_keywords,
        flags=re.IGNORECASE
    )
    match_abstract = re.search(
        r"\n.*?A\s*B\s*S?\s*T\s*R\s*A\s*C\s*T.*?\n", text_after_keywords, flags=re.IGNORECASE
    )

    # Tentukan batas akhir yang paling awal ditemukan
    end_positions = [m.start() for m in [match_paren, match_heading, match_abstract] if m]
    end_position = min(end_positions) if end_positions else None

    # Potong keyword
    keywords = text_after_keywords[:end_position] if end_position is not None else text_after_keywords

    # Normalisasi sederhana
    keywords = re.sub(r"-\n", "", keywords)               # gabung kata terpotong
    keywords = re.sub(r"\b([A-Z])\s+([A-Z])\b", r"\1\2", keywords)  # T HE -> THE
    keywords = re.sub(r"\s+", " ", keywords).strip()
    keywords = re.sub(r"(?i)\b(?:Key\s*words|Keywords|Index\s*Terms)[\s—:-]*", "", keywords, count=1).strip()

    return keywords
