import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime

DATABASE_NAME = "users.db"

def _migrate_db(cursor):
    """Checks the database schema and adds the is_admin column if it's missing."""
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'is_admin' not in columns:
        print("INFO: Database schema is outdated. Adding 'is_admin' column to users table...")
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            print("INFO: 'is_admin' column added successfully.")
        except sqlite3.OperationalError as e:
            print(f"ERROR: Could not add 'is_admin' column: {e}")


def init_db():
    """Initializes the database, creates tables, and adds a default admin if none exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    _migrate_db(c)

    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            filename TEXT,
            analysis_type TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    c.execute("SELECT * FROM users WHERE is_admin = 1")
    if not c.fetchone():
        default_admin_user = 'admin'
        default_admin_pass = 'admin123'
        hashed_password = generate_password_hash(default_admin_pass)
        c.execute("SELECT id FROM users WHERE username = ?", (default_admin_user,))
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        (default_admin_user, hashed_password, 1))
        else: 
            c.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (default_admin_user,))
        
        print("---")
        print("Default admin user created/verified.")
        print(f"Username: {default_admin_user}")
        print(f"Password: {default_admin_pass}")
        print("---")

    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)", (username, password))
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_id(username):
    user = get_user(username)
    return user['id'] if user else None

def check_if_admin(username):
    user = get_user(username)
    return bool(user['is_admin']) if user and 'is_admin' in user.keys() else False

def add_analysis_record(user_id, filename, analysis_type, prediction, confidence):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    current_timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO analysis_history (user_id, filename, analysis_type, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, filename, analysis_type, prediction, confidence, current_timestamp))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM analysis_history WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    history = c.fetchall()
    conn.close()
    return history

def delete_analysis_record(record_id):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def get_user_analysis_count(user_id):
    """Gets the total number of analyses performed by a specific user."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]
        return count if count else 0

def get_total_user_count():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count

def get_total_analysis_count():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM analysis_history")
    count = c.fetchone()[0]
    conn.close()
    return count

def get_all_users():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin FROM users ORDER BY username")
    users = c.fetchall()
    conn.close()
    return users

def delete_user_and_history(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM analysis_history WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Database error during user deletion: {e}")
    finally:
        conn.close()