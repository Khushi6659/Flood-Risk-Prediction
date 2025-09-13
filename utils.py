import sqlite3

DB_NAME = "flood_app.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    # User inputs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            latitude REAL,
            longitude REAL,
            rainfall REAL,
            temperature REAL,
            humidity REAL,
            river_discharge REAL,
            water_level REAL,
            elevation REAL,
            land_cover TEXT,
            soil_type TEXT,
            population_density REAL,
            infrastructure TEXT,
            historical_floods REAL
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_input(username, input_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_inputs (
            username, latitude, longitude, rainfall, temperature, humidity, river_discharge,
            water_level, elevation, land_cover, soil_type, population_density, infrastructure, historical_floods
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        username,
        input_data["Latitude"],
        input_data["Longitude"],
        input_data["Rainfall"],
        input_data["Temperature"],
        input_data["Humidity"],
        input_data["River Discharge"],
        input_data["Water Level"],
        input_data["Elevation"],
        input_data["Land Cover"],
        input_data["Soil Type"],
        input_data["Population Density"],
        input_data["Infrastructure"],
        input_data["Historical Floods"]
    ))
    conn.commit()
    conn.close()
