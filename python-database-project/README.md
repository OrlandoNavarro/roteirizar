# Python Database Project

Este projeto é uma aplicação em Python que permite salvar informações em um banco de dados. A estrutura do projeto é organizada em diferentes módulos, cada um responsável por uma parte específica da aplicação.

## Estrutura do Projeto

```
python-database-project
├── src
│   ├── main.py                # Ponto de entrada da aplicação
│   ├── database
│   │   ├── connection.py      # Estabelece a conexão com o banco de dados
│   │   └── models.py          # Define os modelos que representam as tabelas do banco
│   ├── services
│   │   └── data_service.py     # Contém a lógica para salvar dados no banco
│   └── utils
│       └── helpers.py         # Funções auxiliares para validação e formatação
├── requirements.txt           # Dependências do projeto
├── .env                       # Variáveis de ambiente para configuração
└── README.md                  # Documentação do projeto
```

## Instalação

1. Clone o repositório:
   ```
   git clone <URL_DO_REPOSITORIO>
   cd python-database-project
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows use `venv\Scripts\activate`
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente no arquivo `.env` com as credenciais do banco de dados.

## Uso

Para executar a aplicação, utilize o seguinte comando:
```
python src/main.py
```

A aplicação irá inicializar a conexão com o banco de dados e chamar os serviços necessários para salvar as informações. Certifique-se de que o banco de dados está acessível e que as credenciais estão corretas.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.