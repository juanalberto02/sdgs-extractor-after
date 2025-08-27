# extraction.py (public API: process_sql_text)
import re
import pandas as pd

# =========================
# Regex: unwrap semua wrapper TITLE(-ABS)/ABS/AUTHKEY
# =========================
# Tambah/ubah konstanta regex di dekat import
_UNWRAP_QUOTED_RE = re.compile(
    r'\b(?:TITLE\s*-\s*ABS|TITLE|ABS|AUTHKEY|KEY)\s*\(\s*([\'"])(.*?)\1\s*\)',
    flags=re.IGNORECASE | re.DOTALL
)
_WRAPPER_NAME_BEFORE_PAREN_RE = re.compile(
    r'\b(?:TITLE\s*-\s*ABS|TITLE|ABS|AUTHKEY|KEY)\s*\(',
    flags=re.IGNORECASE
)

def _unwrap_wrappers(s: str) -> str:
    """Hilangkan wrapper TITLE/ABS/AUTHKEY/KEY baik yang ber-kutip maupun tidak.
       - Case 1: WRAP("...")  -> ...
       - Case 2: WRAP( ... )  -> ( ... )   (label dihapus, kurung dipertahankan)
    """
    if not isinstance(s, str):
        return s
    out = s
    # 1) Unwrap yang pakai kutip (ulang sampai habis)
    prev = None
    while out != prev:
        prev = out
        out = _UNWRAP_QUOTED_RE.sub(r'\2', out)
    # 2) Hapus label wrapper sebelum '(' untuk kasus tanpa kutip / bersarang
    out = _WRAPPER_NAME_BEFORE_PAREN_RE.sub('(', out)
    return out

# =========================
# Helpers: Cleaning & Parentheses
# =========================
def clean_rule(rule: str) -> str:
    """Hilangkan wrapper pada rule tunggal."""
    return _unwrap_wrappers(rule)

def remove_title_authkey(text: str) -> str:
    """Hilangkan wrapper pada potongan teks."""
    return _unwrap_wrappers(text)

def remove_extra_parentheses(text: str) -> str:
    """Rapikan spasi & hapus kurung berlebih di tepi."""
    if not isinstance(text, str):
        return text
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'^[(\s]+|[\s)]+$', '', text.strip())

def balance_paren(text: str):
    """Seimbangkan jumlah '(' dan ')' dengan menambah pasangannya di ujung/awal."""
    if not isinstance(text, str):
        return text
    open_paren = text.count('(')
    close_paren = text.count(')')
    if open_paren > close_paren:
        text += ')' * (open_paren - close_paren)
    elif close_paren > open_paren:
        text = '(' * (close_paren - open_paren) + text
    return text

def clean_wrapper_functions(text: str) -> str:
    """Hilangkan token wrapper tersisa dan tanda petik longgar."""
    if not isinstance(text, str):
        return text
    # Buang token TITLE/ABS/AUTHKEY yang tersisa (tanpa kurung) + kutip
    text = re.sub(r'\b(TITLE|ABS|AUTHKEY)\b\s*-?\s*', '', text, flags=re.IGNORECASE)
    return text.replace('"', '').replace("'", '')

def conditional_strip_parentheses(text: str) -> str:
    """Hapus kurung yang tidak perlu jika di dalamnya tidak ada AND/OR."""
    def replace_once(match):
        inner = match.group(1).strip()
        if re.search(r'\bAND\b|\bOR\b', inner, re.IGNORECASE):
            return f'({inner})'
        else:
            return inner
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r'\(([^()]+)\)', replace_once, text)
    return text

def strip_outer_parentheses(text: str) -> str:
    """Strip kurung terluar berlebihan bila seimbang."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    while text.startswith('(') and text.endswith(')'):
        inner = text[1:-1].strip()
        if inner.count('(') == inner.count(')'):
            text = inner
        else:
            break
    return text

def normalize_parentheses(text: str):
    """Hapus ')' nyasar dan tutup semua '(' yang belum tertutup."""
    if not isinstance(text, str):
        return text
    out = []
    bal = 0
    for ch in text:
        if ch == '(':
            bal += 1
            out.append(ch)
        elif ch == ')':
            if bal > 0:
                bal -= 1
                out.append(ch)
            else:
                continue
        else:
            out.append(ch)
    if bal > 0:
        out.append(')' * bal)
    return ''.join(out)

def extract_jenis(text: str) -> str:
    """Deteksi kemunculan TITLE/ABS/AUTHKEY/KEY pada inc_raw (kolom 'jenis')."""
    if not isinstance(text, str):
        return ''
    found = set()
    if re.search(r'\bTITLE\b', text, re.IGNORECASE):
        found.add('title')
    if re.search(r'\bABS\b', text, re.IGNORECASE):
        found.add('abs')
    if re.search(r'\bAUTHKEY\b', text, re.IGNORECASE) or re.search(r'\bKEY\b', text, re.IGNORECASE):
        # anggap KEY sebagai authkey (tetap konsisten dengan skema lama)
        found.add('authkey')
    return ', '.join(sorted(found)) if found else ''

# =========================
# AND NOT Extractions
# =========================
def extract_all_andnot_blocks(text: str):
    """Pisahkan inclusion dan exclusions berdasarkan 'AND NOT (...)'."""
    if not isinstance(text, str):
        return "", []
    inclusion = ""
    exclusions = []
    i = 0
    n = len(text)
    while i < n:
        m = re.search(r'\bAND\s+NOT\b', text[i:], re.IGNORECASE)
        if not m:
            inclusion += text[i:]
            break
        andnot_idx = i + m.start()
        inclusion += text[i:andnot_idx]
        i = andnot_idx + len(m.group())
        while i < n and text[i] != '(':
            i += 1
        if i >= n:
            break
        start = i
        level = 1
        i += 1
        while i < n and level > 0:
            if text[i] == '(':
                level += 1
            elif text[i] == ')':
                level -= 1
            i += 1
        exclusions.append(text[start + 1:i - 1].strip())
    return inclusion.strip(), exclusions

# =========================
# Parser kurung paling luar (fraction/query block)
# =========================
def _extract_query_blocks(sql_text: str):
    queries = []
    start_idx = None
    paren_level = 0
    for i, ch in enumerate(sql_text):
        if ch == '(':
            if paren_level == 0:
                start_idx = i
            paren_level += 1
        elif ch == ')':
            paren_level -= 1
            if paren_level == 0 and start_idx is not None:
                queries.append(sql_text[start_idx:i + 1].strip())
                start_idx = None
    return queries

# =========================
# Split bantuan
# =========================
def split_rules(block: str):
    """
    Split aturan dengan pola ') OR (' (outer-level style), lalu bersihkan wrapper.
    Return: (raw_list, clean_list)
    """
    if not isinstance(block, str):
        return [], []
    block = normalize_parentheses(balance_paren(block))
    parts = [remove_extra_parentheses(p) for p in re.split(r'\)\s+OR\s*\(', block) if p.strip()]
    raw, cleaned = [], []
    for r in parts:
        r_norm = remove_extra_parentheses(r)
        # Bersihkan wrapper dulu, lalu hapus token sisa + kutip
        c = clean_wrapper_functions(remove_extra_parentheses(_unwrap_wrappers(r_norm)))
        raw.append(r_norm)
        cleaned.append(c)
    return raw, cleaned

def clean_inc_exc_raw(text: str) -> str:
    """Bersihkan wrapper + token sisa + kutip dan rapikan spasi (tanpa ubah kurung)."""
    if not isinstance(text, str):
        return text
    text = _unwrap_wrappers(text)
    # buang token sisa (TITLE/ABS/AUTHKEY/KEY) yang berdiri sendiri + kutip
    text = re.sub(r'\b(TITLE|ABS|AUTHKEY|KEY)\b\s*-?\s*', '', text, flags=re.IGNORECASE)
    text = text.replace('"', '').replace("'", '').replace('|', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def split_by_top_level_or(expr: str):
    """Pisah expr pada token 'OR' hanya ketika depth kurung == 0."""
    if not isinstance(expr, str):
        return [expr]
    s = expr.strip()
    parts, buf = [], []
    depth = 0
    i, n = 0, len(s)
    def flush():
        part = ''.join(buf).strip()
        if part:
            parts.append(part)
        buf.clear()
    while i < n:
        ch = s[i]
        if ch == '(':
            depth += 1
            buf.append(ch); i += 1; continue
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch); i += 1; continue
        if depth == 0 and ch.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            word = s[i:j]
            if word.upper() == 'OR':
                prev_ok = (i == 0) or (not s[i-1].isalpha())
                next_ok = (j == n) or (not s[j].isalpha())
                if prev_ok and next_ok:
                    flush(); i = j; continue
            buf.append(word); i = j; continue
        buf.append(ch); i += 1
    flush()
    return parts if parts else [s]


# === NEW: Boolean tokenizer + parser + distributor (AND > OR precedence) ===
_BOOL_SPLIT_RE = re.compile(r'\(|\)|\bAND\b|\bOR\b', flags=re.IGNORECASE)

def _tokenize_bool(s: str):
    if not isinstance(s, str):
        return []
    tokens = []
    pos = 0
    for m in _BOOL_SPLIT_RE.finditer(s):
        if m.start() > pos:
            chunk = s[pos:m.start()].strip()
            if chunk:
                tokens.append(("TEXT", chunk))
        op = m.group()
        if op == '(':
            tokens.append(("LP", '('))
        elif op == ')':
            tokens.append(("RP", ')'))
        else:
            tokens.append(("OP", op.upper()))
        pos = m.end()
    tail = s[pos:].strip()
    if tail:
        tokens.append(("TEXT", tail))
    return tokens

def _parse_factor(tokens, i):
    # factor := TEXT | '(' expr ')'
    if i >= len(tokens):
        return {"type": "TEXT", "text": ""}, i
    t, v = tokens[i]
    if t == "LP":
        node, j = _parse_or(tokens, i+1)
        if j < len(tokens) and tokens[j][0] == "RP":
            return node, j+1
        return node, j  # toleran bila kurung tutup hilang
    elif t == "TEXT":
        return {"type": "TEXT", "text": v}, i+1
    return {"type": "TEXT", "text": ""}, i+1

def _parse_and(tokens, i):
    # and := factor ('AND' factor)*
    node, i = _parse_factor(tokens, i)
    children = [node]
    while i < len(tokens) and tokens[i] == ("OP", "AND"):
        rhs, i = _parse_factor(tokens, i+1)
        children.append(rhs)
    if len(children) == 1:
        return children[0], i
    return {"type": "AND", "children": children}, i

def _parse_or(tokens, i=0):
    # or := and ('OR' and)*
    node, i = _parse_and(tokens, i)
    children = [node]
    while i < len(tokens) and tokens[i] == ("OP", "OR"):
        rhs, i = _parse_and(tokens, i+1)
        children.append(rhs)
    if len(children) == 1:
        return children[0], i
    return {"type": "OR", "children": children}, i

def _dnf_expand(node):
    # Kembalikan list string; setiap item adalah satu conjunct (gabungan AND)
    if node.get("type") == "TEXT":
        txt = node.get("text", "").strip()
        return [txt] if txt else []
    if node.get("type") == "AND":
        parts = [ _dnf_expand(ch) for ch in node["children"] ]
        # Cartesian product lalu join dengan AND
        out = ['']
        for lst in parts:
            new_out = []
            for base in out:
                for piece in lst:
                    if not base:
                        new_out.append(piece)
                    elif not piece:
                        new_out.append(base)
                    else:
                        new_out.append(f"{base} AND {piece}")
            out = new_out
        return [o for o in out if o.strip()]
    if node.get("type") == "OR":
        out = []
        for ch in node["children"]:
            out.extend(_dnf_expand(ch))
        return [o for o in out if o.strip()]
    # fallback
    return []

def expand_or_recursively(expr: str):
    """
    Distribusikan OR di dalam kurung:
      A AND (B OR C) => [ "A AND B", "A AND C" ]
    Berlaku rekursif, precedence AND > OR.
    """
    tokens = _tokenize_bool(expr)
    if not tokens:
        return [expr.strip()] if isinstance(expr, str) else [expr]
    ast, _ = _parse_or(tokens, 0)
    parts = _dnf_expand(ast)
    cleaned = []
    for p in parts if parts else [expr]:
        p = re.sub(r"\s+", " ", p).strip()
        p = strip_outer_parentheses(balance_paren(p))
        p = normalize_parentheses(p)
        cleaned.append(p)
    return [c for c in cleaned if c]


# =========================
# Public API (tetap)
# =========================
def process_sql_text(text: str, sdgs_input: int) -> pd.DataFrame:
    """
    Return dataframe: sdg, no, inc_raw, exc_raw, inc_clean, exc_clean, jenis
    """
    queries = _extract_query_blocks(text)
    final_rows = []
    row_no = 1

    for query in queries:
        query = balance_paren(query)
        inclusion, exclusions = extract_all_andnot_blocks(query)

        inc_raw_list, inc_clean_list = split_rules(inclusion)

        exc_raw_all, exc_clean_all = [], []
        for exc_block in exclusions:
            er, ec = split_rules(exc_block)
            exc_raw_all.extend(er)
            exc_clean_all.extend(ec)

        exc_raw_joined = " OR ".join(exc_raw_all)
        exc_clean_joined = " OR ".join(exc_clean_all)

        # --- NEW: Hapus SEMUA tanda kurung di exclusion ---
        if exc_clean_joined:
            exc_clean_joined = re.sub(r'[()]', '', exc_clean_joined)  # hapus semua kurung
            exc_clean_joined = re.sub(r'\s+', ' ', exc_clean_joined).strip()  # rapikan spasi


        for raw, clean in zip(inc_raw_list, inc_clean_list):
            raw_bal = balance_paren(raw)
            jenis_val = extract_jenis(raw_bal)
            final_rows.append({
                "sdg": int(sdgs_input),
                "no": str(row_no),
                "inc_raw": remove_extra_parentheses(raw_bal),
                "exc_raw": balance_paren(exc_raw_joined) if exc_raw_joined else "",
                "inc_clean": strip_outer_parentheses(
                    normalize_parentheses(re.sub(r'\s+', ' ', clean).strip())
                ),
                "exc_clean": strip_outer_parentheses(
                    normalize_parentheses(re.sub(r'\s+', ' ', exc_clean_joined).strip())
                ),
                "jenis": jenis_val
            })
            row_no += 1

    df = pd.DataFrame(final_rows, columns=[
        "sdg", "no", "inc_raw", "exc_raw", "inc_clean", "exc_clean", "jenis"
    ])

    # Post-fix: balance & trim
    df["inc_raw"] = df["inc_raw"].apply(balance_paren)
    df["exc_raw"] = df["exc_raw"].apply(balance_paren)
    for c in ["inc_raw", "exc_raw", "inc_clean", "exc_clean"]:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Hapus kurung yang tidak perlu di inc_clean (misal welfare AND (economic status))
    df["inc_clean"] = df["inc_clean"].apply(conditional_strip_parentheses)

    # Split OR level-atas pada inc_clean (opsional, kalau mau granular)
    # Split OR (termasuk yang bersarang) pada inc_clean → distribusi penuh
    split_rows = []
    for _, row in df.iterrows():
        inc_val = str(row["inc_clean"]).strip()
        parts = expand_or_recursively(inc_val)
        if len(parts) > 1:
            for idx, part in enumerate(parts, 1):
                new_row = row.to_dict()
                new_row["no"] = f'{row["no"]}.{idx}'
                new_row["inc_clean"] = part
                split_rows.append(new_row)
        else:
            split_rows.append(row.to_dict())

    df = pd.DataFrame(split_rows, columns=[
        "sdg", "no", "inc_raw", "exc_raw", "inc_clean", "exc_clean", "jenis"
    ]).reset_index(drop=True)


    return df

