# main.py
from fastapi import FastAPI, Request, Form, UploadFile, Path
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import mysql.connector
import pandas as pd
import shutil, os
from extraction import process_sql_text
from detection import detect_from_pdf_with_rules  
import json
import datetime
import pymysql
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fastapi import Query
from fastapi.responses import StreamingResponse
import io


app = FastAPI()
load_dotenv()

def save_deteksi_history(username, result):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        ssl={
            "ca": "DigiCertGlobalRootCA.crt.pem"
        }
    )
    kw = result.get("keywords", "")
    # Simpan keywords sebagai JSON string jika list/dict
    if isinstance(kw, (list, dict)):
        kw = json.dumps(kw, ensure_ascii=False)

    tr = result.get("top_rules", [])
    if not isinstance(tr, list):
        tr = []

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO deteksi_history (username, title, abstract, keywords, top_rules, deteksi_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        username,
        result.get("title", ""),
        result.get("abstract", ""),
        kw,
        json.dumps(tr, ensure_ascii=False),  # selalu simpan list sebagai JSON
        datetime.now()
    ))
    conn.commit()
    conn.close()


def fetch_rules_from_mysql():
    import pandas as pd
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ca": "DigiCertGlobalRootCA.crt.pem"}
    )
    cursor = conn.cursor()
    cursor.execute("SELECT sdg, no, inc_raw, exc_raw, inc_clean, exc_clean, jenis FROM extraction")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        df = pd.DataFrame(rows)
        # Kompatibilitas dengan detection.py lama:
        df = df.rename(columns={
            "inc_clean": "inc",
            "exc_clean": "exc"
        })
    else:
        df = pd.DataFrame(columns=["sdg","no","inc_raw","exc_raw","inc","exc","jenis"])

    print("Fetch rules (first 3):", df.head(3))
    print("Shape:", df.shape)
    return df





app.add_middleware(SessionMiddleware, secret_key="supersecret")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")

def get_user_from_db(username: str):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        ssl={
            "ca": "DigiCertGlobalRootCA.crt.pem"
        }
    )


    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    conn.close()
    return result

def save_to_mysql(df):
    import numpy as np
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        ssl={ "ca": "DigiCertGlobalRootCA.crt.pem" }
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extraction (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sdg INT,
            no VARCHAR(20),
            inc_raw TEXT,
            exc_raw TEXT,
            inc_clean TEXT,
            exc_clean TEXT,
            jenis VARCHAR(64)
        )
    """)

    # Hapus baris header jika ada (edge case upload CSV sebagai text)
    if len(df) and (df.iloc[0].astype(str).values == df.columns.astype(str)).all():
        df = df.iloc[1:]

    # Casting ringan
    df = df.replace({np.nan: None})
    if "sdg" in df:
        df["sdg"] = pd.to_numeric(df["sdg"], errors="coerce").fillna(0).astype(int)
    for col in ["no","inc_raw","exc_raw","inc_clean","exc_clean","jenis"]:
        if col in df:
            df[col] = df[col].astype(str)

    for _, r in df.iterrows():
        cursor.execute("""
            INSERT INTO extraction (sdg, no, inc_raw, exc_raw, inc_clean, exc_clean, jenis)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            r.get("sdg"), r.get("no"),
            r.get("inc_raw"), r.get("exc_raw"),
            r.get("inc_clean"), r.get("exc_clean"),
            r.get("jenis"),
        ))
    conn.commit()
    conn.close()


def fetch_from_mysql(sdg_input=None):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        ssl={ "ca": "DigiCertGlobalRootCA.crt.pem" }
    )
    cursor = conn.cursor()
    query = "SELECT id, sdg, no, inc_raw, exc_raw, inc_clean, exc_clean, jenis FROM extraction"
    params = ()
    if sdg_input:
        query += " WHERE sdg=%s"
        params = (sdg_input,)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["id","sdg","no","inc_raw","exc_raw","inc_clean","exc_clean","jenis"])
    return df

def get_analytics_summary():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ca": "DigiCertGlobalRootCA.crt.pem"}
    )
    cursor = conn.cursor()

    # Total input
    cursor.execute("SELECT COUNT(*) as total FROM deteksi_history")
    total_input = cursor.fetchone()['total']

    # Daily input (WIB-safe)
    today = datetime.utcnow() + timedelta(hours=7)
    start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    cursor.execute("SELECT COUNT(*) as daily FROM deteksi_history WHERE deteksi_date BETWEEN %s AND %s", (start_of_day, end_of_day))
    daily_input = cursor.fetchone()['daily']

    # Avg similarity dari top_rules JSON
    cursor.execute("SELECT top_rules FROM deteksi_history")
    rows = cursor.fetchall()
    all_scores = []

    for row in rows:
        try:
            top_rules = json.loads(row['top_rules'])
            # Validasi bahwa ini list of dict, dan ada similarity numerik
            for rule in top_rules:
                if isinstance(rule, dict):
                    similarity = rule.get("similarity")
                    if isinstance(similarity, (int, float)):
                        all_scores.append(similarity)
        except Exception:
            continue

    avg_similarity = round(sum(all_scores) / len(all_scores) * 100, 2) if all_scores else 0.0

    # Monthly chart (8 bulan terakhir)
    cursor.execute("""
        SELECT DATE_FORMAT(deteksi_date, '%Y-%m') AS month, COUNT(*) as count
        FROM deteksi_history
        WHERE deteksi_date >= DATE_SUB(CURDATE(), INTERVAL 7 MONTH)
        GROUP BY month
        ORDER BY month
    """)
    monthly = cursor.fetchall()
    monthly_labels = [row['month'] for row in monthly]
    monthly_data = [row['count'] for row in monthly]

    # Weekly Chart (last 8 weeks), week_start = tanggal Senin
    cursor.execute("""
        SELECT 
            DATE_FORMAT(
                DATE_SUB(deteksi_date, INTERVAL WEEKDAY(deteksi_date) DAY),
                '%Y-%m-%d'
            ) AS week_start,
            COUNT(*) AS count
        FROM deteksi_history
        WHERE deteksi_date >= CURDATE() - INTERVAL 7 WEEK
        GROUP BY week_start
        ORDER BY week_start
    """)
    weekly = cursor.fetchall()
    weekly_labels = [row['week_start'] for row in weekly]
    weekly_data = [row['count'] for row in weekly]
    
    # Daily Chart (last 7 days)
    cursor.execute("""
        SELECT DATE(deteksi_date) AS date, COUNT(*) as count
        FROM deteksi_history
        WHERE deteksi_date >= CURDATE() - INTERVAL 6 DAY
        GROUP BY date
        ORDER BY date
                   """)

    daily = cursor.fetchall()
    daily_labels = [row['date'].strftime("%Y-%m-%d") for row in daily]
    daily_data = [row['count'] for row in daily]

    # Hitung jumlah kemunculan setiap SDG dari top_rules
    sdg_count = [0] * 17  # SDG 1 sampai 17

    for row in rows:
        try:
            top_rules = json.loads(row['top_rules'])
            for rule in top_rules:
                if isinstance(rule, dict):
                    sdg_index = rule.get("sdg")  # Misal: 0 untuk SDG 1
                    if isinstance(sdg_index, int) and 0 <= sdg_index < 17:
                        sdg_count[sdg_index] += 1
        except Exception:
            continue
    # Mapping SDG to pillars
    sdg_to_pillars = {
        1: ["Social", "Economic"],
        2: ["Social", "Economic", "Environmental"],
        3: ["Social"],
        4: ["Social"],
        5: ["Social", "Governance"],
        6: ["Environmental", "Social"],
        7: ["Environmental", "Economic"],
        8: ["Economic"],
        9: ["Economic", "Governance"],
        10: ["Social", "Governance"],
        11: ["Environmental", "Social", "Governance"],
        12: ["Environmental"],
        13: ["Environmental"],
        14: ["Environmental"],
        15: ["Environmental"],
        16: ["Governance"],
        17: ["Governance", "Economic"]
    }

    # Hitung jumlah pilar dari semua top_rules
    cursor.execute("SELECT top_rules FROM deteksi_history")
    rows = cursor.fetchall()
    pillar_counter = {"Social": 0, "Economic": 0, "Environmental": 0, "Governance": 0}

    for row in rows:
        try:
            top_rules = json.loads(row['top_rules'])
            for rule in top_rules:
                sdg = rule.get("sdg")
                if isinstance(sdg, int) and sdg in sdg_to_pillars:
                    for pillar in sdg_to_pillars[sdg]:
                        pillar_counter[pillar] += 1
        except Exception:
            continue
    
    pillar_distribution = [
        {"label": pillar, "count": count}
        for pillar, count in pillar_counter.items()
        if count > 0
    ]

    conn.close()

    return {
        "total_input": total_input,
        "daily_input": daily_input,
        "avg_similarity": avg_similarity,
        "chart_data": {
            "monthly": {"labels": monthly_labels, "data": monthly_data},
            "weekly":  {"labels": weekly_labels,  "data": weekly_data},
            "daily":   {"labels": daily_labels,   "data": daily_data}
        },
        "sdg_count": sdg_count,
        "pillar_distribution": pillar_counter
        }


def fetch_user_history(username: str, q: str = "", offset: int = 0, limit: int = 20):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ca": "DigiCertGlobalRootCA.crt.pem"}
    )
    cur = conn.cursor()
    where = "WHERE username=%s"
    params = [username]
    if q:
        where += " AND title LIKE %s"
        params.append(f"%{q}%")

    # total
    cur.execute(f"SELECT COUNT(*) AS n FROM deteksi_history {where}", params)
    total = cur.fetchone()["n"]

    # data
    cur.execute(
        f"""SELECT id, title, abstract, keywords, top_rules, deteksi_date
            FROM deteksi_history
            {where}
            ORDER BY deteksi_date DESC
            LIMIT %s OFFSET %s""",
        params + [limit, offset]
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        # parse keywords (bisa JSON string atau plain string)
        kw_raw = r.get("keywords") or ""
        try:
            kw_parsed = json.loads(kw_raw) if isinstance(kw_raw, str) and kw_raw.strip().startswith(("[", "{")) else kw_raw
        except Exception:
            kw_parsed = kw_raw
        # normalisasi keywords → list atau string
        if isinstance(kw_parsed, list):
            keywords_norm = kw_parsed
        elif isinstance(kw_parsed, str):
            # biarkan string; front-end bisa split comma bila perlu
            keywords_norm = kw_parsed
        else:
            keywords_norm = ""

        # parse top_rules
        try:
            rules = json.loads(r.get("top_rules") or "[]")
        except Exception:
            rules = []

        # normalisasi tiap rule (pastikan sdg:int, similarity:float)
        norm_rules = []
        for rr in rules or []:
            if not isinstance(rr, dict):
                continue
            try:
                sdg_val = int(rr.get("sdg", 0))
            except Exception:
                sdg_val = 0
            try:
                sim_val = float(rr.get("similarity", 0.0))
            except Exception:
                sim_val = 0.0
            # gabungkan lagi field lain
            norm_rules.append({
                **rr,
                "sdg": sdg_val,
                "similarity": sim_val,
            })

        items.append({
            "id": r.get("id"),
            "title": r.get("title") or "",
            "abstract": r.get("abstract") or "",
            "keywords": keywords_norm,     # bisa list atau string
            "deteksi_date": r["deteksi_date"].strftime("%Y-%m-%d %H:%M") if r.get("deteksi_date") else "",
            "top_rules": norm_rules        # ← KIRIM LENGKAP, bukan hanya top-3
        })

    return items, total


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, page: int = 1, per_page: int = 10, q: str = ""):
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)

    username = request.session.get("user")
    page = max(1, int(page))
    per_page = max(1, min(50, int(per_page)))
    offset = (page - 1) * per_page

    items, total = fetch_user_history(username, q=q.strip(), offset=offset, limit=per_page)
    start = offset
    end = min(offset + len(items), total)

    # pilih default tampilan (item pertama) agar panel kanan langsung terisi
    active_result = {}
    if items:
        first = items[0]
        active_result = {
            "title": first.get("title", ""),
            "abstract": first.get("abstract", ""),
            "keywords": first.get("keywords", ""),
            "top_rules": first.get("top_rules", []),
        }

    return templates.TemplateResponse("history.html", {
        "request": request,
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "start": start,
        "end": end,
        "q": q.strip(),
        "active_result": active_result,   # ← penting untuk hydrate
    })

@app.post("/history/use/{row_id}")
def history_use(request: Request, row_id: int):
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)

    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ca": "DigiCertGlobalRootCA.crt.pem"}
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT title, abstract, keywords, top_rules FROM deteksi_history WHERE id=%s AND username=%s",
        (row_id, request.session.get("user"))
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return RedirectResponse("/history", status_code=303)

    try:
        rules = json.loads(row["top_rules"]) if row.get("top_rules") else []
    except Exception:
        rules = []

    # normalisasi
    norm_rules = []
    for rr in rules or []:
        try:
            sdg_val = int(rr.get("sdg", 0))
        except Exception:
            sdg_val = 0
        try:
            sim_val = float(rr.get("similarity", 0.0))
        except Exception:
            sim_val = 0.0
        norm_rules.append({"sdg": sdg_val, "similarity": sim_val,
                           **{k:v for k,v in rr.items() if k not in ("sdg","similarity")}})

    request.session["deteksi_result"] = {
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "keywords": row.get("keywords", ""),
        "top_rules": norm_rules
    }
    return RedirectResponse("/deteksi", status_code=303)

@app.post("/history/clear")
def history_clear(request: Request):
    # wajib login sebagai user
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)

    username = request.session.get("user")

    # koneksi DB (samakan dengan fungsi fetch/save kamu)
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ca": "DigiCertGlobalRootCA.crt.pem"}
    )
    cur = conn.cursor()
    # hapus hanya history user ini
    cur.execute("DELETE FROM deteksi_history WHERE username=%s", (username,))
    conn.commit()
    conn.close()

    # kembali ke /history
    return RedirectResponse(url="/history", status_code=303)


@app.get("/ekstraksi", response_class=HTMLResponse)
def ekstraksi_page(request: Request):
    if not request.session.get("user") or request.session.get("role") != "admin":
        return RedirectResponse("/login", status_code=303)
    df = fetch_from_mysql()
    rows_html = ""
    if not df.empty:
        for _, row in df.iterrows():
            rows_html += f"""
            <tr>
                <td style="text-align:center;font-size:12px;">{row['sdg']}</td>
                <td style="text-align:center;font-size:12px;">{row['no']}</td>
                <td style="text-align:center;font-size:12px;">{row['jenis']}</td>
                <td style="font-size:12px;">{row['inc_clean']}</td>
                <td style="font-size:12px;">{row['exc_clean']}</td>
                <td style="height: 48px; padding: 0;">
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                        <form method="post" action="/ekstraksi/delete/{row['id']}" style="display:inline;">
                            <button type="submit" class="btn btn-danger rounded-circle"
                                    style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;padding:0;"
                                    onclick="return confirm('Delete this row?');">
                                <i class="bi bi-trash" style="font-size: 1rem;"></i>
                            </button>
                        </form>
                    </div>
                </td>
            </tr>
            """
    return templates.TemplateResponse("ekstraksi_sdg.html", {
        "request": request,
        "error": "",
        "table_rows": rows_html
    })


@app.post("/ekstraksi", response_class=HTMLResponse)
async def ekstraksi_upload(request: Request, file: UploadFile = Form(...), sdgs_input: int = Form(...)):
    if not request.session.get("user") or request.session.get("role") != "admin":
        return RedirectResponse("/login", status_code=303)

    if not (file.filename.endswith(".sql") or file.filename.endswith(".txt")):
        return templates.TemplateResponse("ekstraksi_sdg.html", {
            "request": request,
            "error": "Hanya format .sql dan .txt yang diperbolehkan.",
            "table_rows": ""
        })

    contents = await file.read()
    try:
        text = contents.decode("utf-8")
        df = process_sql_text(text, int(sdgs_input))
        save_to_mysql(df)

        df_show = fetch_from_mysql()
        rows_html = ""
        for _, row in df_show.iterrows():
            rows_html += f"""
            <tr>
                <td style="text-align:center;font-size:12px;">{row['sdg']}</td>
                <td style="text-align:center;font-size:12px;">{row['no']}</td>
                <td style="text-align:center;font-size:12px;">{row['jenis']}</td>
                <td style="font-size:12px;">{row['inc_clean']}</td>
                <td style="font-size:12px;">{row['exc_clean']}</td>
                <td style="height: 48px; padding: 0;">
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                        <form method="post" action="/ekstraksi/delete/{row['id']}" style="display:inline;">
                            <button type="submit" class="btn btn-danger rounded-circle"
                                    style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;padding:0;"
                                    onclick="return confirm('Delete this row?');">
                                <i class="bi bi-trash" style="font-size: 1rem;"></i>
                            </button>
                        </form>
                    </div>
                </td>
            </tr>
            """
        return templates.TemplateResponse("ekstraksi_sdg.html", {
            "request": request,
            "error": "",
            "table_rows": rows_html
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        # Selalu coba tampilkan tabel, walaupun kosong
        df_show = fetch_from_mysql()
        rows_html = ""
        if df_show is not None and not df_show.empty:
            for _, row in df_show.iterrows():
                rows_html += f"""
                <tr>
                    <td style="text-align:center;font-size:12px;">{row['sdg']}</td>
                    <td style="text-align:center;font-size:12px;">{row['no']}</td>
                    <td style="text-align:center;font-size:12px;">{row['jenis']}</td>
                    <td style="font-size:12px;">{row['inc_clean']}</td>
                    <td style="font-size:12px;">{row['exc_clean']}</td>
                    <td style="height: 48px; padding: 0;">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                            <form method="post" action="/ekstraksi/delete/{row['id']}" style="display:inline;">
                                <button type="submit" class="btn btn-danger rounded-circle"
                                        style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;padding:0;"
                                        onclick="return confirm('Delete this row?');">
                                    <i class="bi bi-trash" style="font-size: 1rem;"></i>
                                </button>
                            </form>
                        </div>
                    </td>
                </tr>
                """
        return templates.TemplateResponse("ekstraksi_sdg.html", {
            "request": request,
            "error": f"Gagal memproses file: {str(e)}",
            "table_rows": rows_html
        })


# Tambahkan endpoint untuk hapus semua data
@app.post("/ekstraksi/delete_all")
async def delete_all(request: Request):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        ssl={
            "ca": "DigiCertGlobalRootCA.crt.pem"
        }
    )


    cursor = conn.cursor()
    cursor.execute("DELETE FROM extraction")
    conn.commit()
    conn.close()
    return RedirectResponse("/ekstraksi", status_code=303)

@app.get("/ekstraksi/download")
def download_extraction():
    df = fetch_from_mysql()
    if df.empty:
        return RedirectResponse("/ekstraksi", status_code=303)

    # Simpan ke CSV di memory
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    headers = {
        'Content-Disposition': 'attachment; filename="extraction_data.csv"'
    }
    return StreamingResponse(iter([stream.getvalue()]),
                             media_type="text/csv",
                             headers=headers)

@app.get("/analytics", response_class=HTMLResponse)
def article_page(request: Request):
    if not request.session.get("user") or request.session.get("role") != "admin":
        return RedirectResponse("/login", status_code=303)

    summary = get_analytics_summary()

    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "total_input": summary["total_input"],
        "daily_input": summary["daily_input"],
        "avg_similarity": summary["avg_similarity"],
        "chart_data": json.dumps(summary["chart_data"]),
        "sdg_count": summary["sdg_count"],
        "pillar_distribution": summary["pillar_distribution"]

    })


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user_from_db(username)
    if not user or user["password"] != password:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Login gagal"})
    request.session["user"] = user["username"]
    request.session["role"] = user["role"]

    # reset hasil deteksi di session agar halaman deteksi mulai fresh
    request.session.pop("deteksi_result", None)

    if user["role"] == "user":
        # tandai sekali setelah login; front-end kamu sudah paham flag/param ini
        request.session["afterLoginReset"] = "1"
        return RedirectResponse("/deteksi?reset=1", status_code=303)
    else:
        return RedirectResponse("/ekstraksi", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

@app.get("/deteksi", response_class=HTMLResponse)
def deteksi_page(request: Request, reset: int = Query(0)):
    # Pastikan user & role benar
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)

    # Reset sekali saat baru login (datang dengan ?reset=1)
    if reset == 1:
        # buang hasil lama agar UI mulai dari upload lagi
        request.session.pop("deteksi_result", None)
        # bersihkan flag sekali-pakai jika kamu memakainya
        request.session.pop("afterLoginReset", None)

        # render halaman kosong (upload dulu)
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": "",
            "abstract": "",
            "keywords": "",
            "top_rules": [],
            "all_rules": [],          # <-- penting: selalu ada di template
            "error": ""
        })

    # Ambil hasil deteksi terakhir dari session (jika ada)
    result = request.session.get("deteksi_result")
    if result:
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": result.get("title", ""),
            "abstract": result.get("abstract", ""),
            "keywords": result.get("keywords", ""),
            "top_rules": result.get("top_rules", []),
            "all_rules": [],          # <-- tidak disimpan di session, biarkan kosong
            "error": ""
        })

    # Jika tidak ada hasil, tampilkan kosong (upload dulu)
    return templates.TemplateResponse("deteksi_sdg.html", {
        "request": request,
        "title": "",
        "abstract": "",
        "keywords": "",
        "top_rules": [],
        "all_rules": [],              # <-- penting: selalu ada
        "error": ""
    })


@app.post("/deteksi", response_class=HTMLResponse)
async def deteksi_upload(
    request: Request,
    pdf_file: UploadFile = Form(...)
):
    # Cek login
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)

    # Simpan file sementara
    os.makedirs("tmp", exist_ok=True)
    pdf_path = f"tmp/{pdf_file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())

    try:
        # Ambil rules, jalankan deteksi
        rules_df = fetch_rules_from_mysql()
        result = detect_from_pdf_with_rules(pdf_path, rules_df)

        # ===== Anti ERR_RESPONSE_HEADERS_TOO_BIG =====
        # Simpan versi ramping ke session (tanpa all_rules yang besar)
        slim = {
            "title": result.get("title", ""),
            "abstract": result.get("abstract", ""),
            "keywords": result.get("keywords", ""),
            "top_rules": result.get("top_rules", []),
        }
        request.session["deteksi_result"] = slim

        # Simpan riwayat deteksi (boleh pakai objek penuh)
        username = request.session.get("user")
        try:
            save_deteksi_history(username, result)
        except Exception:
            # Jangan ganggu UX kalau logging history gagal
            pass

        # Render halaman dengan all_rules (langsung dari result, TANPA disimpan di session)
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": slim["title"],
            "abstract": slim["abstract"],
            "keywords": slim["keywords"],
            "top_rules": slim["top_rules"],
            "all_rules": result.get("all_rules", []),  # <-- dipakai untuk tombol See All
            "error": ""
        })

    except Exception as e:
        # Bersihkan session pada error
        request.session["deteksi_result"] = None
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": "",
            "abstract": "",
            "keywords": "",
            "top_rules": [],
            "all_rules": [],   # <-- tetap kirim agar JS aman
            "error": f"Error: {str(e)}"
        })

@app.get("/articles/{article_name}", response_class=HTMLResponse)
def read_article(request: Request, article_name: str):
    return templates.TemplateResponse(f"articles/{article_name}", {"request": request})

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/debug-db")
def debug_db():
    df = fetch_from_mysql()
    print(df.head())
    return {"n_rows": len(df), "columns": list(df.columns)}

@app.get("/cek_ekstraksi")
def cek_ekstraksi():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={ "ca": "DigiCertGlobalRootCA.crt.pem" }
    )
    cursor = conn.cursor()
    cursor.execute("SELECT DATABASE() as db, COUNT(*) as n, (SELECT COUNT(*) FROM users) as n_users FROM extraction;")
    result = cursor.fetchone()
    cursor.execute("SELECT * FROM extraction LIMIT 5;")
    rows = cursor.fetchall()
    conn.close()
    return {"meta": result, "sample_rows": rows}
