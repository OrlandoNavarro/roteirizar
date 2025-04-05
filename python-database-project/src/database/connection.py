from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def create_connection():
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    
    return engine.connect()