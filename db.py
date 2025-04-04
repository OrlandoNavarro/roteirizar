import sqlite3
import datetime

DB_NAME = "uploads.db"

def initialize_db():
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            file_name TEXT,
            file_content BLOB,
            upload_date TEXT
        )
    ''')
    # Cria tabela para coordenadas, se não existir
    cur.execute('''
        CREATE TABLE IF NOT EXISTS coordenadas (
            endereco TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL
        )
    ''')
    con.commit()
    con.close()

def save_upload(username, file_name, file_content):
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    upload_date = datetime.datetime.now().isoformat()
    cur.execute('''
        INSERT INTO uploads (username, file_name, file_content, upload_date)
        VALUES (?, ?, ?, ?)
    ''', (username, file_name, file_content, upload_date))
    con.commit()
    con.close()

def get_saved_coordinates(endereco):
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    cur.execute("SELECT latitude, longitude FROM coordenadas WHERE endereco=?", (endereco,))
    result = cur.fetchone()
    con.close()
    return result

def save_coordinates(endereco, latitude, longitude):
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO coordenadas (endereco, latitude, longitude) VALUES (?, ?, ?)",
                (endereco, latitude, longitude))
    con.commit()
    con.close()

if __name__ == '__main__':
    initialize_db()