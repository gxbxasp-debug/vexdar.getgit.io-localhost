"""Vexdar v2: chatbot + transfer file dua arah + pembaca file.

Jalankan lokal : python main.py  ->  http://localhost:8080
Discloud       : TYPE=site, port 8080, host 0.0.0.0
"""
import os
import re
import json
import time
import hmac
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, send_file, jsonify, g
from werkzeug.utils import secure_filename
from openai import (
    OpenAI, RateLimitError, AuthenticationError, NotFoundError,
    BadRequestError, APIConnectionError, APITimeoutError, InternalServerError,
)

load_dotenv()

# ---------- Konfigurasi (diisi lewat file .env) ----------
BASE = Path(__file__).parent.resolve()
ROOT = (BASE / "files").resolve()          # satu-satunya folder yang boleh diakses
ROOT.mkdir(exist_ok=True)

API_KEY = os.environ.get("VEXDAR_API_KEY", "")
TOKEN = os.environ.get("VEXDAR_TOKEN", "")
BASE_URL = os.environ.get("VEXDAR_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("VEXDAR_MODEL", "")
MODEL_CADANGAN = os.environ.get("VEXDAR_MODEL_CADANGAN", "")
DAFTAR_MODEL = [m for m in (MODEL, MODEL_CADANGAN) if m]

SYSTEM = (
    "Kamu adalah Vexdar, asisten AI yang ramah dan ringkas. "
    "Jawab dalam bahasa Indonesia santai kecuali diminta lain. "
    "Kamu punya tools untuk file di penyimpanan Vexdar: list_files, read_file (txt, md, csv, json, log, pdf, docx) "
    "dan create_text_file (membuat file teks baru yang nanti bisa diunduh pengguna ke HP). "
    "Kamu tidak bisa melihat isi gambar. "
    "Jika file panjang, baca bertahap dengan parameter start. "
    "Jangan mengarang isi file; jika file kosong atau tidak terbaca, katakan apa adanya. "
    "Setelah membuat file, sebutkan nama filenya. Jika tidak tahu, katakan tidak tahu."
)

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".log"}
EXT_BOLEH = TEXT_EXT | {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
TULIS_EXT = {".txt", ".md", ".csv", ".json"}   # jenis file yang boleh dibuat oleh AI

MAX_RIWAYAT = 12            # jumlah pesan terakhir yang dikirim ke AI
MAX_UPLOAD_MB = 10          # ukuran maksimal satu file
MAX_TOTAL_MB = 200          # total isi folder files/
BATAS_PER_MENIT = 30
CHUNK = 8000                # karakter per bacaan file oleh AI
PREVIEW_MAX = 20000         # karakter maksimal pada tampilan "Lihat"
MAX_HALAMAN_PDF = 200
MAX_TULIS = 200_000         # karakter maksimal file buatan AI

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "kosong", timeout=90)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------- Keamanan & utilitas nama file ----------
def nama_aman(raw: str) -> str:
    """Bersihkan nama file (tanpa folder, tanpa karakter aneh), pertahankan ekstensi."""
    raw = (raw or "").strip()
    ext = Path(raw).suffix.lower()
    stem = secure_filename(Path(raw).stem) or "file"
    return stem + ext


def aman(nama: str) -> Path:
    """Path file di dalam folder files/; tolak apa pun di luar folder itu."""
    nama = nama_aman(nama)
    p = (ROOT / nama).resolve()
    if p.parent != ROOT:
        raise ValueError("path tidak diizinkan")
    return p


def nama_unik(nama: str) -> Path:
    """Kalau nama sudah dipakai, tambahkan -1, -2, ... (tidak menimpa file lama)."""
    p = ROOT / nama
    stem, ext = p.stem, p.suffix
    i = 1
    while p.exists():
        p = ROOT / f"{stem}-{i}{ext}"
        i += 1
    return p


def semua_file():
    return [f for f in ROOT.iterdir() if f.is_file() and not f.name.startswith(".")]


def total_byte() -> int:
    return sum(f.stat().st_size for f in semua_file())


def terotorisasi() -> bool:
    t = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(TOKEN) and hmac.compare_digest(t.encode(), TOKEN.encode())


_waktu = []


def kena_limit() -> bool:
    sekarang = time.time()
    _waktu[:] = [x for x in _waktu if sekarang - x < 60]
    _waktu.append(sekarang)
    return len(_waktu) > BATAS_PER_MENIT


@app.before_request
def jaga():
    if request.path in ("/", "/health") or request.path.startswith("/static/"):
        return None
    if not terotorisasi():
        return jsonify(error="Token salah atau kosong."), 401
    if kena_limit():
        return jsonify(error="Terlalu banyak permintaan. Tunggu sebentar."), 429
    return None


@app.errorhandler(413)
def terlalu_besar(_):
    return jsonify(error=f"File terlalu besar (maksimal {MAX_UPLOAD_MB} MB)."), 413


@app.errorhandler(Exception)
def semua_error(e):
    print("ERROR:", repr(e))
    return jsonify(error="Terjadi kesalahan di server."), 500


# ---------- Pembaca file ----------
@lru_cache(maxsize=8)
def _ekstrak(path_str: str, mtime: float) -> str:
    p = Path(path_str)
    ext = p.suffix.lower()
    if ext in TEXT_EXT:
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        r = PdfReader(str(p))
        if r.is_encrypted:
            raise ValueError("PDF terkunci dengan kata sandi.")
        hal = []
        for i, pg in enumerate(r.pages[:MAX_HALAMAN_PDF]):
            hal.append(f"[Halaman {i + 1}]\n" + (pg.extract_text() or ""))
        return "\n\n".join(hal)
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        bagian = [x.text for x in d.paragraphs if x.text.strip()]
        for t in d.tables:
            for row in t.rows:
                bagian.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(bagian)
    raise ValueError("Jenis file ini tidak bisa dibaca sebagai teks.")


def ekstrak(p: Path) -> str:
    try:
        teks = _ekstrak(str(p), p.stat().st_mtime)
    except ImportError:
        raise ValueError("Library pembaca file belum terpasang (pypdf / python-docx).")
    if p.suffix.lower() == ".pdf" and not any(c.isalnum() for c in re.sub(r"\[Halaman \d+\]", "", teks)):
        raise ValueError("PDF ini tampaknya hasil scan (gambar), tidak ada teks yang bisa dibaca.")
    return teks


# ---------- Tools yang bisa dipakai AI ----------
def tool_list_files():
    data = sorted(semua_file(), key=lambda f: f.stat().st_mtime, reverse=True)
    if not data:
        return "Penyimpanan kosong."
    return json.dumps([{"name": f.name, "bytes": f.stat().st_size} for f in data], ensure_ascii=False)


def tool_read_file(name="", start=0):
    p = aman(name)
    if not p.is_file():
        return f"File {name} tidak ditemukan."
    teks = ekstrak(p)
    total = len(teks)
    try:
        start = max(0, int(start))
    except (TypeError, ValueError):
        start = 0
    if start >= total:
        return f"[{p.name}] sudah habis dibaca (total {total} karakter)."
    akhir = min(total, start + CHUNK)
    kepala = f"[{p.name} | karakter {start}-{akhir} dari {total}]\n"
    if akhir < total:
        ekor = f"\n[Masih ada lanjutan. Panggil read_file lagi dengan start={akhir}]"
    else:
        ekor = "\n[Akhir file]"
    return kepala + teks[start:akhir] + ekor


def tool_create_text_file(name="", content=""):
    nama = nama_aman(name)
    if Path(nama).suffix.lower() not in TULIS_EXT:
        return "Hanya boleh membuat file .txt, .md, .csv, atau .json."
    if not isinstance(content, str) or not content.strip():
        return "Isi file kosong."
    if len(content) > MAX_TULIS:
        return f"Isi terlalu panjang (maksimal {MAX_TULIS} karakter)."
    data = content.encode("utf-8")
    if total_byte() + len(data) > MAX_TOTAL_MB * 1024 * 1024:
        return "Penyimpanan penuh."
    p = nama_unik(nama)
    p.write_bytes(data)
    g.dibuat.append(p.name)
    return f"File dibuat: {p.name} ({len(data)} byte). Pengguna bisa mengunduhnya dari panel File."


TOOLS_IMPL = {
    "list_files": lambda **_: tool_list_files(),
    "read_file": lambda **kw: tool_read_file(kw.get("name", ""), kw.get("start", 0)),
    "create_text_file": lambda **kw: tool_create_text_file(kw.get("name", ""), kw.get("content", "")),
}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Tampilkan daftar file (nama dan ukuran) di penyimpanan Vexdar.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Baca isi file teks, PDF, atau DOCX. File panjang dibaca bertahap lewat start.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "nama file persis seperti di daftar"},
            "start": {"type": "integer", "description": "posisi karakter awal (default 0)"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "create_text_file",
        "description": "Buat file teks baru (.txt, .md, .csv, .json) di penyimpanan; tidak menimpa file lama.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "nama file, contoh: ringkasan.md"},
            "content": {"type": "string", "description": "isi file"}},
            "required": ["name", "content"]}}},
]


# ---------- Halaman ----------
@app.get("/")
def home():
    return send_file(BASE / "index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, model=MODEL or None)


# ---------- Chat ----------
def panggil_ai(msgs, pakai_tools):
    """Coba model utama; kalau penyedia sedang bermasalah, coba model cadangan."""
    terakhir = None
    for model in DAFTAR_MODEL:
        try:
            kw = {"tools": TOOLS_SPEC} if pakai_tools else {}
            return client.chat.completions.create(
                model=model, messages=msgs, max_tokens=4000, **kw)
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            terakhir = e
    raise terakhir


@app.post("/chat")
def chat():
    if not API_KEY or not DAFTAR_MODEL:
        return jsonify(error="API key atau model belum diisi di file .env."), 500

    data = request.get_json(silent=True) or {}
    riwayat = [
        {"role": m["role"], "content": m["content"]}
        for m in data.get("messages", [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ][-MAX_RIWAYAT:]
    if not riwayat:
        return jsonify(error="Pesan kosong."), 400

    g.dibuat = []
    msgs = [{"role": "system", "content": SYSTEM}] + riwayat
    pakai_tools = True

    for _ in range(8):
        try:
            r = panggil_ai(msgs, pakai_tools)
        except BadRequestError:
            if pakai_tools:            # model tidak mendukung tools -> coba tanpa tools
                pakai_tools = False
                continue
            return jsonify(error="Permintaan ditolak oleh penyedia AI."), 502
        except AuthenticationError:
            return jsonify(error="API key ditolak. Periksa VEXDAR_API_KEY."), 502
        except NotFoundError:
            return jsonify(error="Model tidak ditemukan. Periksa VEXDAR_MODEL."), 502
        except RateLimitError:
            return jsonify(error="Kuota atau batas penyedia AI habis. Coba lagi nanti."), 429
        except (APIConnectionError, APITimeoutError, InternalServerError):
            return jsonify(error="Penyedia AI sedang tidak bisa dihubungi. Coba lagi sebentar lagi."), 504

        m = r.choices[0].message
        if not m.tool_calls:
            return jsonify(reply=m.content or "(tidak ada jawaban)", files=g.dibuat)

        msgs.append({
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in m.tool_calls
            ],
        })
        for c in m.tool_calls:
            try:
                fn = TOOLS_IMPL[c.function.name]
                hasil = fn(**json.loads(c.function.arguments or "{}"))
            except Exception as e:
                hasil = f"Error: {e}"
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": str(hasil)})

    return jsonify(reply="Terlalu banyak langkah. Coba perintah yang lebih sederhana.", files=g.dibuat)


# ---------- Transfer file (HP <-> server) ----------
@app.post("/upload")
def upload():
    daftar = request.files.getlist("file")
    if not daftar:
        return jsonify(error="Tidak ada file."), 400
    tersimpan, galat = [], []
    for f in daftar:
        nama = nama_aman(f.filename)
        if Path(nama).suffix.lower() not in EXT_BOLEH:
            galat.append(f"{f.filename}: jenis file tidak diizinkan")
            continue
        isi = f.read()
        if total_byte() + len(isi) > MAX_TOTAL_MB * 1024 * 1024:
            galat.append(f"{f.filename}: penyimpanan penuh")
            continue
        p = nama_unik(nama)
        p.write_bytes(isi)
        tersimpan.append(p.name)
    if not tersimpan:
        return jsonify(error="; ".join(galat) or "Gagal menyimpan."), 400
    return jsonify(saved=tersimpan, errors=galat)


@app.get("/files")
def files():
    data = sorted(semua_file(), key=lambda f: f.stat().st_mtime, reverse=True)
    return jsonify([
        {"name": f.name, "size": f.stat().st_size, "mtime": int(f.stat().st_mtime)}
        for f in data
    ])


@app.get("/preview/<path:name>")
def preview(name):
    p = aman(name)
    if not p.is_file():
        return jsonify(error="File tidak ditemukan."), 404
    try:
        teks = ekstrak(p)
    except Exception as e:
        return jsonify(error=str(e)), 422
    return jsonify(text=teks[:PREVIEW_MAX], terpotong=len(teks) > PREVIEW_MAX, total=len(teks))


@app.get("/download/<path:name>")
def download(name):
    p = aman(name)
    if not p.is_file():
        return jsonify(error="File tidak ditemukan."), 404
    return send_file(p, as_attachment=True)


@app.delete("/files/<path:name>")
def hapus(name):
    p = aman(name)
    if not p.is_file():
        return jsonify(error="File tidak ditemukan."), 404
    p.unlink()
    return jsonify(deleted=p.name)


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    print(f"Vexdar berjalan di http://0.0.0.0:{port}")
    serve(app, host="0.0.0.0", port=port, threads=8)