import os
from dotenv import load_dotenv
from database.connection import create_connection
from services.data_service import DataService

def main():
    load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
    connection = create_connection()  # Estabelece a conexão com o banco de dados

    if connection:
        data_service = DataService(connection)
        # Aqui você pode chamar métodos do data_service para salvar informações
        # Exemplo: data_service.save_data(data)

if __name__ == "__main__":
    main()