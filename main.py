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


app = FastAPI()
load_dotenv()

def save_deteksi_history(username, result):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO deteksi_history (username, title, abstract, keywords, top_rules, deteksi_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        username,
        result["title"],
        result["abstract"],
        result["keywords"],
        json.dumps(result["top_rules"]),  # Simpan list sebagai string JSON
        datetime.datetime.now()
    ))
    conn.commit()
    conn.close()


def fetch_rules_from_mysql():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    query = "SELECT sdg, no, inc_raw, inc, exc_raw FROM ekstraksi"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


app.add_middleware(SessionMiddleware, secret_key="supersecret")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")

def get_user_from_db(username: str):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    conn.close()
    return result

def save_to_mysql(df):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ekstraksi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sdg INT,
            fraction INT,
            no INT,
            inc_raw TEXT,
            inc TEXT,
            exc_raw TEXT,
            exc TEXT
        )
    """)
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO ekstraksi (sdg, fraction, no, inc_raw, inc, exc_raw, exc)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (row['sdg'], row['fraction'], row['no'], row['inc_raw'], row['inc'], row['exc_raw'], row['exc']))
    conn.commit()
    conn.close()

def fetch_from_mysql(sdg_input=None):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    query = "SELECT id, sdg, inc_raw, inc FROM ekstraksi"
    params = ()
    if sdg_input:
        query += " WHERE sdg=%s"
        params = (sdg_input,)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

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
                <td style="text-align: center;font-size: 12px;">{row['sdg']}</td>
                <td style="text-align: center;font-size: 12px;">{row['inc_raw']}</td>
                <td style="text-align: center;font-size: 12px;">{row['inc']}</td>
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

    if not file.filename.endswith(".sql"):
        return templates.TemplateResponse("ekstraksi_sdg.html", {
            "request": request,
            "error": "Hanya file .sql yang diperbolehkan.",
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
                <td style="text-align: center;font-size: 12px;">{row['sdg']}</td>
                <td style="text-align: center;font-size: 12px;">{row['inc_raw']}</td>
                <td style="text-align: center;font-size: 12px;">{row['inc']}</td>
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
        return templates.TemplateResponse("ekstraksi_sdg.html", {
            "request": request,
            "error": f"Gagal memproses file: {str(e)}",
            "table_rows": ""
        })

# Tambahkan endpoint untuk hapus satu baris
@app.post("/ekstraksi/delete/{row_id}")
async def delete_row(request: Request, row_id: int = Path(...)):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()
    cursor.execute("DELETE FROM ekstraksi WHERE id = %s", (row_id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/ekstraksi", status_code=303)

# Tambahkan endpoint untuk hapus semua data
@app.post("/ekstraksi/delete_all")
async def delete_all(request: Request):
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()
    cursor.execute("DELETE FROM ekstraksi")
    conn.commit()
    conn.close()
    return RedirectResponse("/ekstraksi", status_code=303)

@app.get("/analytics", response_class=HTMLResponse)
def article_page(request: Request):
    # Cek apakah sudah login dan role admin
    if not request.session.get("user") or request.session.get("role") != "admin":
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("analytics.html", {"request": request})


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
    return RedirectResponse("/ekstraksi" if user["role"] == "admin" else "/deteksi", status_code=303)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

@app.get("/deteksi", response_class=HTMLResponse)
def deteksi_page(request: Request):
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)
    
    # Ambil hasil deteksi terakhir dari session (jika ada)
    result = request.session.get("deteksi_result")
    if result:
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": result.get("title", ""),
            "abstract": result.get("abstract", ""),
            "keywords": result.get("keywords", ""),
            "top_rules": result.get("top_rules", []),
            "error": ""
        })
    # Jika tidak ada, tampilkan kosong
    return templates.TemplateResponse("deteksi_sdg.html", {
        "request": request,
        "title": "",
        "abstract": "",
        "keywords": "",
        "top_rules": [],
        "error": ""
    })

@app.post("/deteksi", response_class=HTMLResponse)
async def deteksi_upload(
    request: Request, 
    pdf_file: UploadFile = Form(...)
):
    if not request.session.get("user") or request.session.get("role") != "user":
        return RedirectResponse("/login", status_code=303)
    os.makedirs("tmp", exist_ok=True)
    pdf_path = f"tmp/{pdf_file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())
    try:
        rules_df = fetch_rules_from_mysql()
        result = detect_from_pdf_with_rules(pdf_path, rules_df)
        request.session["deteksi_result"] = result
        
        # Simpan riwayat deteksi
        username = request.session.get("user")
        save_deteksi_history(username, result)
        
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": result["title"],
            "abstract": result["abstract"],
            "keywords": result["keywords"],
            "top_rules": result["top_rules"],
            "error": ""
        })
    except Exception as e:
        request.session["deteksi_result"] = None
        return templates.TemplateResponse("deteksi_sdg.html", {
            "request": request,
            "title": "",
            "abstract": "",
            "keywords": "",
            "top_rules": [],
            "error": f"Error: {str(e)}"
        })

@app.get("/article", response_class=HTMLResponse)
def article_page(request: Request):
    return templates.TemplateResponse("article.html", {"request": request})



@app.get("/articles/{article_name}", response_class=HTMLResponse)
def read_article(request: Request, article_name: str):
    return templates.TemplateResponse(f"articles/{article_name}", {"request": request})

@app.get("/sdgs_detail/{sdg_name}", response_class=HTMLResponse)
def read_sdg_detail(request: Request, sdg_name: str):
    return templates.TemplateResponse(f"sdgs_detail/{sdg_name}", {"request": request})

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


def initialize_users_table():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()

    # Buat tabel users jika belum ada
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(100),
            role VARCHAR(10)
        )
    """)

    # Cek apakah user admin sudah ada, jika belum buat
    cursor.execute("SELECT COUNT(*) AS count FROM users WHERE username='admin'")
    result = cursor.fetchone()
    if result["count"] == 0:
        cursor.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')")

    # Cek apakah user biasa sudah ada, jika belum buat
    cursor.execute("SELECT COUNT(*) AS count FROM users WHERE username='user'")
    result = cursor.fetchone()
    if result["count"] == 0:
        cursor.execute("INSERT INTO users (username, password, role) VALUES ('user', 'user123', 'user')")

    conn.commit()
    conn.close()

def initialize_ekstraksi_table():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ekstraksi (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sdg INT,
            fraction INT,
            no INT,
            inc_raw TEXT,
            inc TEXT,
            exc_raw TEXT,
            exc TEXT
        )
    """)

    conn.commit()
    conn.close()

def initialize_deteksi_history_table():
    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),  # Use environment variable for host
        user=os.getenv("DB_USER"),  # Use environment variable for user
        password=os.getenv("DB_PASSWORD"),  # Use environment variable for password
        database=os.getenv("DB_NAME"),  # Use environment variable for database name
        port=int(os.getenv("DB_PORT", 11564)),  # Default to 11564 if DB_PORT is not set
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deteksi_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50),
            title TEXT,
            abstract TEXT,
            keywords TEXT,
            top_rules TEXT,
            deteksi_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

initialize_users_table()
initialize_ekstraksi_table()
initialize_deteksi_history_table()