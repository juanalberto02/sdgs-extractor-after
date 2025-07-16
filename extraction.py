# extraction.py
import re
import pandas as pd

def process_sql_text(text, sdgs_input):
    def clean_rule(rule):
        rule = re.sub(r'TITLE\s*-\s*ABS\s*\(\s*"([^"]+)"\s*\)', r'\1', rule, flags=re.IGNORECASE)
        rule = re.sub(r'AUTHKEY\s*\(\s*"([^"]+)"\s*\)', r'\1', rule, flags=re.IGNORECASE)
        return rule

    def remove_title_authkey(text):
        text = re.sub(r'TITLE\s*-\s*ABS\s*\(\s*"([^"]+)"\s*\)', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'AUTHKEY\s*\(\s*"([^"]+)"\s*\)', r'\1', text, flags=re.IGNORECASE)
        return text

    def remove_extra_parentheses(text):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'^[(\s]+|[\s)]+$', '', text.strip())

    def balance_parentheses(text):
        open_paren = text.count('(')
        close_paren = text.count(')')
        if open_paren > close_paren:
            text += ')' * (open_paren - close_paren)
        elif close_paren > open_paren:
            text = '(' * (close_paren - open_paren) + text
        return text

    def extract_rules(text):
        text = balance_parentheses(text)
        if "AND NOT(" in text:
            parts = text.split("AND NOT(", 1)
            include_part, exclude_part = parts[0].strip(), parts[1].strip().rstrip(")")
        else:
            include_part, exclude_part = text.strip(), ""
        include_part = include_part.rstrip("OR")
        include_raw = [remove_extra_parentheses(rule.strip()) for rule in re.split(r'\)\s+OR\s*\(', include_part) if rule]
        exclude_raw = [remove_extra_parentheses(rule.strip()) for rule in re.split(r'\)\s+OR\s*\(', exclude_part) if rule]
        include_cleaned = [remove_title_authkey(remove_extra_parentheses(clean_rule(rule))) for rule in include_raw]
        exclude_cleaned = [remove_title_authkey(remove_extra_parentheses(clean_rule(rule))) for rule in exclude_raw]
        return include_raw, include_cleaned, exclude_raw, exclude_cleaned

    # --- Ekstraksi Kurung untuk Ambil Query ---
    queries = []
    start_idx = None
    paren_level = 0
    for i, char in enumerate(text):
        if char == '(':
            if paren_level == 0:
                start_idx = i
            paren_level += 1
        elif char == ')':
            paren_level -= 1
            if paren_level == 0 and start_idx is not None:
                query = text[start_idx:i + 1].strip()
                queries.append(query)
                start_idx = None

    # --- Proses Semua Query ---
    all_data = []
    for query in queries:
        inc_raw, inc_clean, exc_raw, exc_clean = extract_rules(query)
        inc_raw = [balance_parentheses(s) for s in inc_raw]
        inc_clean = [balance_parentheses(s) for s in inc_clean]
        exc_raw = [balance_parentheses(s) for s in exc_raw]
        exc_clean = [balance_parentheses(s) for s in exc_clean]
        df_output = format_output(inc_raw, inc_clean, exc_raw, exc_clean, sdgs_input)
        all_data.append(df_output)

    df_final = pd.concat(all_data, ignore_index=True)
    df_final['inc'] = df_final['inc'].apply(remove_title_authkey)
    df_final['exc'] = df_final['exc'].apply(remove_title_authkey)
    df_final['inc'] = df_final['inc'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    df_final['exc'] = df_final['exc'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df_final
