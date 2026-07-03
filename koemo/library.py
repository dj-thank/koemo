"""会議履歴と全文検索（SQLite）。"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import CONFIG_DIR

DB_FILE = CONFIG_DIR / "library.db"


def _connect():
    CONFIG_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    _init(conn)
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _init(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meetings(
            id INTEGER PRIMARY KEY,
            ts TEXT,
            title TEXT,
            date TEXT,
            summary_md TEXT,
            transcript_md TEXT,
            save_dir TEXT,
            duration INT,
            created_at TEXT
        )
    """)
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts
            USING fts5(title, summary_md, transcript_md, content='meetings', content_rowid='id')
        """)
    except sqlite3.Error:
        pass
    conn.commit()


def _rowdicts(rows):
    return [dict(r) for r in rows]


def _meeting_date(ts):
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").date().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def add(title, summary_md, transcript_md, save_dir, duration, ts):
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO meetings(ts, title, date, summary_md, transcript_md, save_dir, duration, created_at)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (ts, title, _meeting_date(ts), summary_md, transcript_md, str(save_dir),
             int(duration or 0), datetime.now().isoformat(timespec="seconds")),
        )
        mid = cur.lastrowid
        try:
            conn.execute(
                "INSERT INTO meetings_fts(rowid, title, summary_md, transcript_md) VALUES(?,?,?,?)",
                (mid, title, summary_md, transcript_md),
            )
        except sqlite3.Error:
            pass
        conn.commit()
        return mid


def recent(limit=50):
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return _rowdicts(rows)


def get(meeting_id):
    with _db() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id=?", (int(meeting_id),)).fetchone()
        return dict(row) if row else None


def search(query, limit=50):
    query = (query or "").strip()
    if not query:
        return recent(limit)
    with _db() as conn:
        try:
            rows = conn.execute(
                """
                SELECT m.* FROM meetings_fts f
                JOIN meetings m ON m.id = f.rowid
                WHERE meetings_fts MATCH ?
                ORDER BY bm25(meetings_fts)
                LIMIT ?
                """,
                (_quote_fts(query), int(limit)),
            ).fetchall()
            if rows:
                return _rowdicts(rows)
        except sqlite3.Error:
            pass
        like = f"%{query}%"
        rows = conn.execute(
            """
            SELECT * FROM meetings
            WHERE title LIKE ? OR summary_md LIKE ? OR transcript_md LIKE ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        ).fetchall()
        return _rowdicts(rows)


def _quote_fts(query):
    parts = [p.replace('"', '""') for p in query.split() if p.strip()]
    if not parts:
        return '""'
    return " AND ".join(f'"{p}"' for p in parts)
