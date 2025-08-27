# detection.py
# -------------------------------------------------------
# Fungsi utama:
#   detect_from_pdf_with_rules(pdf_path, rules_df) -> dict
#   (Output dan field sesuai KONTRAK yang kamu minta)
# -------------------------------------------------------

import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd

from extractors_text import (
    extract_title_full_span,
    extract_abstract_smart,
    extract_text_for_keywords,
    extract_keywords_from_text,
)

# --- SBERT imports ---
import os
import numpy as np
from sentence_transformers import SentenceTransformer
_SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_sbert = None

def _get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(_SBERT_MODEL)
    return _sbert

def _sbert_encode(texts):
    model = _get_sbert()
    # normalize_embeddings=True agar cosine bisa pakai dot
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

def _cosine_dot(u: np.ndarray, v: np.ndarray) -> float:
    # vektor sudah dinormalisasi -> cosine = dot
    return float(np.dot(u, v))


# ===== Konfigurasi =====
ADJACENT_PHRASES_DEFAULT = False  # default non-adjacent untuk token tanpa kutip
MIN_ABSTRACT_WORDS = 25

# ===== Normalisasi & Utils =====
def norm_loose(s: str) -> str:
    """Buang semua non-alfabet untuk pencocokan longgar (emi ssions -> emissions)."""
    return re.sub(r"[^a-z]", "", (s or "").lower())

def norm_words(s: str) -> str:
    """Pertahankan spasi untuk frasa yang butuh adjacency."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def cosine_sim(a: str, b: str) -> float:
    """Cosine similarity sederhana di atas TF kata (lower & alnum)."""
    def bow(x: str) -> Counter:
        toks = re.findall(r"[a-z0-9]+", (x or "").lower())
        return Counter(toks)
    va, vb = bow(a), bow(b)
    if not va or not vb:
        return 0.0
    dot = sum(va[t]*vb.get(t,0) for t in va)
    na = math.sqrt(sum(v*v for v in va.values()))
    nb = math.sqrt(sum(v*v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot/(na*nb)

# ===== Token/Frasa Presence =====
def _phrase_present(hay: str, needle: str, adjacency: bool) -> bool:
    if not needle:
        return False
    if adjacency:
        # adjacent phrase → pakai norm_words (spasi dipertahankan)
        return norm_words(needle) in norm_words(hay)
    else:
        # non-adjacent → split kata, cocokkan per-kata pakai norm_loose
        parts = [p for p in re.split(r"\s+", needle.strip()) if p]
        n_hay = norm_loose(hay)
        for p in parts:
            if norm_loose(p) not in n_hay:
                return False
        return True

def _match_token(hay: str, token: str) -> bool:
    """
    - Jika token di-quote → frasa adjacent
    - Jika tidak → per-kata (non-adjacent), toleran spasi/simbol
    """
    t = (token or "").strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        return _phrase_present(hay, t[1:-1], adjacency=True)
    return _phrase_present(hay, t, adjacency=ADJACENT_PHRASES_DEFAULT)

# ===== Parser Query Boolean Sederhana =====
TOKEN_RE = re.compile(r'\s*(\(|\)|"[^"]+"|AND|OR|NOT|[^\s()]+)\s*', re.IGNORECASE)

class Node: ...
@dataclass
class Word(Node):
    text: str
@dataclass
class Not(Node):
    child: Node
@dataclass
class And(Node):
    left: Node
    right: Node
@dataclass
class Or(Node):
    left: Node
    right: Node

def tokenize(q: str) -> List[str]:
    out, i = [], 0
    while i < len(q):
        m = TOKEN_RE.match(q, i)
        if not m:
            break
        out.append(m.group(1))
        i = m.end()
    return out

class Parser:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, val: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of query")
        if val and tok.upper() != val:
            raise ValueError(f"Expected {val}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_term()
        while True:
            tok = self.peek()
            if tok and tok.upper() == "OR":
                self.eat()
                node = Or(node, self.parse_term())
            else:
                break
        return node

    def parse_term(self) -> Node:
        node = self.parse_factor()
        while True:
            tok = self.peek()
            if tok and tok.upper() == "AND":
                self.eat()
                node = And(node, self.parse_factor())
            else:
                break
        return node

    def parse_factor(self) -> Node:
        tok = self.peek()
        if tok and tok.upper() == "NOT":
            self.eat()
            return Not(self.parse_factor())
        if tok == "(":
            self.eat("(")
            node = self.parse()
            self.eat(")")
            return node
        return Word(self.eat())

def eval_node(node: Node, hay: str) -> bool:
    if isinstance(node, Word):
        return _match_token(hay, node.text)
    if isinstance(node, Not):
        return not eval_node(node.child, hay)
    if isinstance(node, And):
        return eval_node(node.left, hay) and eval_node(node.right, hay)
    if isinstance(node, Or):
        return eval_node(node.left, hay) or eval_node(node.right, hay)
    return False

# ===== Helpers utk deteksi =====

def _plain_terms(query: str) -> List[str]:
    """Ambil token 'kata/frasa' saja (tanpa operator dan kurung) untuk coverage."""
    toks = tokenize(query or "")
    out = []
    for t in toks:
        if t.upper() in {"AND","OR","NOT"} or t in {"(",")"}:
            continue
        out.append(t)
    return out

def _jenis_fields(jenis: str) -> List[str]:
    """
    Petakan 'jenis' ke field list.
    Contoh nilai umum: 'title', 'abstract', 'keywords', 'abs, title', 'abs, title, keywords', dst.
    Default → semua.
    """
    j = (jenis or "").lower()
    fields = set()
    if "title" in j: fields.add("title")
    if "abstract" in j or "abs" in j: fields.add("abstract")
    if "keyword" in j: fields.add("keywords")
    if not fields:
        fields = {"title", "abstract", "keywords"}
    return list(fields)

def _per_field_found_missing(query: str, title: str, abstract: str, keywords: str) -> Dict[str, Dict[str, List[str]]]:
    toks = _plain_terms(query)
    out = {"title": {"found": [], "missing": []},
           "abstract": {"found": [], "missing": []},
           "keywords": {"found": [], "missing": []}}
    for t in toks:
        (out["title"]["found"] if _match_token(title, t) else out["title"]["missing"]).append(t)
        (out["abstract"]["found"] if _match_token(abstract, t) else out["abstract"]["missing"]).append(t)
        (out["keywords"]["found"] if _match_token(keywords, t) else out["keywords"]["missing"]).append(t)
    return out

def _check_rule_against_text(query: str, scope_text: str) -> Tuple[List[str], List[str], float, int, int]:
    """Kembalikan found_terms, missing_terms, coverage [0..1], present_cnt, total_cnt."""
    terms = _plain_terms(query)
    if not terms:
        return [], [], 0.0, 0, 0
    found, miss = [], []
    for t in terms:
        (_match_token(scope_text, t) and found.append(t)) or miss.append(t) if not _match_token(scope_text, t) else None
    present_cnt = len(found)
    total_cnt = len(terms)
    cov = present_cnt / total_cnt if total_cnt else 0.0
    return found, miss, cov, present_cnt, total_cnt

def _exclusion_found(exc: str, jenis: str, title: str, abstract: str, keywords: str) -> Tuple[bool, List[str]]:
    """Cek exclusion terms hadir di scope sesuai jenis."""
    if not exc or not exc.strip():
        return False, []
    fields = _jenis_fields(jenis)
    hay = []
    if "title" in fields: hay.append(title or "")
    if "abstract" in fields: hay.append(abstract or "")
    if "keywords" in fields: hay.append(keywords or "")
    hay = " ".join(hay)
    toks = _plain_terms(exc)
    hit = []
    for t in toks:
        if _match_token(hay, t):
            hit.append(t)
    return (len(hit) > 0), hit

def _required_badge(inc: str, jenis: str, title: str, abstract: str, keywords: str) -> str:
    """String ringkas untuk UI lama."""
    fields = _jenis_fields(jenis)
    hay = []
    if "title" in fields: hay.append(title or "")
    if "abstract" in fields: hay.append(abstract or "")
    if "keywords" in fields: hay.append(keywords or "")
    hay = " ".join(hay)
    terms = _plain_terms(inc)
    if not terms:
        return "-"
    found = [t for t in terms if _match_token(hay, t)]
    if len(found) == len(terms):
        return "✅ All required words present"
    miss = [t for t in terms if t not in found]
    return "⚠️ Missing: " + ", ".join(miss)

def _similarity_score(a: str, b: str) -> float:
    """Nilai 0..1 (cosine sederhana)."""
    return float(cosine_sim(a or "", b or ""))

# ===== Ekstraksi Fields PDF =====
def _extract_fields(pdf_path: str, debug: bool = False) -> Dict[str, str]:
    title = extract_title_full_span(pdf_path, max_pages=1, debug=debug)
    abstract = extract_abstract_smart(pdf_path, debug=debug)
    if abstract and isinstance(abstract, str) and len(abstract.split()) < MIN_ABSTRACT_WORDS:
        abstract = abstract.strip()

    kw_text = extract_text_for_keywords(pdf_path, max_pages=3)
    keywords = extract_keywords_from_text(kw_text)

    return {
        "title": title or "",
        "abstract": abstract or "",
        "keywords": keywords or "",
    }

# ====== FUNGSI UTAMA — KONTRAK TETAP ======
def detect_from_pdf_with_rules(pdf_path: str, rules_df: pd.DataFrame) -> dict:
    # 1) Extract Title/Abstract/Keywords
    title = extract_title_full_span(pdf_path, max_pages=1, debug=False)
    abstract = extract_abstract_smart(pdf_path)
    kw_text = extract_text_for_keywords(pdf_path, max_pages=3)
    keywords = extract_keywords_from_text(kw_text)

    # 2) Siapkan teks gabungan untuk similarity
    combined = f"{title}. {abstract}. {keywords}"
    # 2a) Precompute SBERT embeddings (hemat: sekali dokumen, batch semua inc)
    #    - Dokumen (gabungan title/abstract/keywords) → 1 embedding
    #    - Semua 'inc' pada rules_df → batch embedding
    inc_list = [str(r.get("inc", "") or "") for _, r in rules_df.iterrows()]
    # encode doc + all inc sekaligus agar 1 panggilan
    sbert_vecs = _sbert_encode([combined] + inc_list)
    doc_vec = sbert_vecs[0]
    inc_vecs = sbert_vecs[1:]

    if rules_df is None or rules_df.empty:
        return {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "top_rules": [],
            "all_rules": [],
        }

    ranked_rows = []
    for idx, (_, r) in enumerate(rules_df.iterrows()):
        sdg   = int(r.get("sdg", 0) or 0)
        no    = str(r.get("no", "") or "")
        inc   = str(r.get("inc", "") or "")
        exc   = str(r.get("exc", "") or "")
        jenis = str(r.get("jenis", "") or "")

        fm = _per_field_found_missing(inc, title, abstract, keywords)

        scope_text_parts = []
        fields = _jenis_fields(jenis)
        if "title" in fields:    scope_text_parts.append(title or "")
        if "abstract" in fields: scope_text_parts.append(abstract or "")
        if "keywords" in fields: scope_text_parts.append(keywords or "")
        scope_text = " ".join(scope_text_parts)

        found_all, miss_all, cov, present_cnt, total_cnt = _check_rule_against_text(inc, scope_text)

        excl_found, excl_terms = _exclusion_found(exc, jenis, title, abstract, keywords)

        req_badge = _required_badge(inc, jenis, title, abstract, keywords)
        unnec_badge = f"🧹 Remove: {', '.join(excl_terms)}" if excl_found else "✅ No unnecessary words found"

        # --- Similarity: BOW (lama) + SBERT (baru) ---
        sim_bow = _similarity_score(combined, inc)
        sim_bert = _cosine_dot(doc_vec, inc_vecs[idx]) if inc_vecs[idx] is not None else 0.0

        entry = {
            "sdg": sdg,
            "no": no,
            "inc": inc,
            "exc": exc,
            "jenis": jenis,

            # sim lama tetap disimpan jika UI lama masih pakai key 'similarity'
            "similarity_bow": float(sim_bow),
            "similarity_bert": float(sim_bert),
            "similarity": float(sim_bert),   # backward-compat: jadikan SBERT default

            "required_words": ("All required words present" if cov == 1.0 else ("Missing: " + ", ".join(miss_all))) if total_cnt else "-",
            "unnecessary_words": (", ".join(excl_terms) if excl_found else "No unnecessary words found"),
            "required_words_badge": req_badge,
            "unnecessary_words_badge": unnec_badge,

            "match_flag": bool(present_cnt == total_cnt and total_cnt > 0),
            "match_present_terms": found_all,
            "match_missing_terms": miss_all,
            "match_total_terms": int(total_cnt),
            "match_present_count": int(present_cnt),
            "match_coverage": float(cov),
            "match_missing_pct": float(1.0 - cov),

            "exclusion_found": bool(excl_found),
            "exclusion_terms_found": excl_terms,

            "present_terms_title":    fm["title"]["found"],
            "missing_terms_title":    fm["title"]["missing"],
            "present_terms_abstract": fm["abstract"]["found"],
            "missing_terms_abstract": fm["abstract"]["missing"],
            "present_terms_keywords": fm["keywords"]["found"],
            "missing_terms_keywords": fm["keywords"]["missing"],
        }
        ranked_rows.append(entry)

    # --- Ranking utama: utamakan match → coverage → similarity_bert
    all_rules = sorted(
        ranked_rows,
        key=lambda x: (x["match_flag"], x["match_coverage"], x["similarity_bert"]),
        reverse=True
    )

    # --- top_rules: per SDG, kalau ada match ambil yang match; jika tidak, ambil similarity_bert tertinggi
    top_rules = []
    by_sdg = {}
    for e in all_rules:
        by_sdg.setdefault(e["sdg"], []).append(e)
    for sdg, items in sorted(by_sdg.items(), key=lambda x: x[0]):
        items_match = [it for it in items if it["match_flag"]]
        pick = (sorted(items_match, key=lambda x: (x["match_coverage"], x["similarity_bert"]), reverse=True)[0]
                if items_match else
                sorted(items, key=lambda x: x["similarity_bert"], reverse=True)[0])
        top_rules.append(pick)


    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "top_rules": top_rules,
        "all_rules": all_rules,
    }
