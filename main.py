import requests
import pandas as pd
import streamlit as st
import sqlite3
import datetime
from sklearn.cluster import KMeans, DBSCAN
import networkx as nx
from itertools import permutations
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import random
from db import initialize_db, save_upload, get_saved_coordinates, save_coordinates
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import geopandas as gpd
from shapely.geometry import Point
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/rota")
def obter_rota():
    return {"rota": ["Ponto A", "Ponto B", "Ponto C"]}

@app.post("/roteirizar")
def roteirizar(pedidos: list, caminhoes: list):
    # Converte os dados recebidos em DataFrames
    pedidos_df = pd.DataFrame(pedidos)
    caminhoes_df = pd.DataFrame(caminhoes)

    # Processa os dados (substitua pelas funções ajustadas)
    pedidos_df = agrupar_por_regiao(pedidos_df)
    dist_matrix = calcular_matriz_distancias(pedidos_df)
    rota_tsp, distancia_tsp = resolver_tsp_ortools(dist_matrix)

    return {
        "rota_tsp": rota_tsp,
        "distancia_tsp": distancia_tsp
    }

# Inicializa o banco de dados (caso ainda não exista)
initialize_db()

# Credenciais padrões
DB_USERNAME = "Orlando"
DB_PASSWORD = "Picole2024@"

# Endereço de partida fixo
endereco_partida = "Avenida Antonio Ortega, 3604 - Pinhal, Cabreúva - SP, São Paulo, Brasil"
# Coordenadas geográficas do endereço de partida
endereco_partida_coords = (-23.0838, -47.1336)  # Exemplo de coordenadas para Cabreúva, SP

# Função para obter coordenadas geográficas de um endereço usando OpenCage
def obter_coordenadas_opencage(endereco):
    try:
        api_key = "6f522c67add14152926990afbe127384"  # Sua chave de API do OpenCage
        url = f"https://api.opencagedata.com/geocode/v1/json?q={endereco}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if 'status' in data and data['status']['code'] == 200 and 'results' in data:
            location = data['results'][0]['geometry']
            return (location['lat'], location['lng'])
        else:
            st.error(f"Não foi possível obter as coordenadas para o endereço: {endereco}. Status: {data.get('status', {}).get('message', 'Desconhecido')}")
            return None
    except Exception as e:
        st.error(f"Erro ao tentar obter as coordenadas: {e}")
        return None

# Função para obter coordenadas com fallback para coordenadas manuais
def obter_coordenadas_com_fallback(endereco):
    # Primeiro, tenta recuperar as coordenadas do banco de dados
    coords = get_saved_coordinates(endereco)
    if coords is not None:
        return coords  # (latitude, longitude)
    
    # Caso não exista no banco, consulta a API
    coords = obter_coordenadas_opencage(endereco)
    if coords is not None:
        # Salva as coordenadas no banco para futuras consultas
        save_coordinates(endereco, coords[0], coords[1])
        return coords
    else:
        return (None, None)

# Função para calcular distância entre dois endereços usando a fórmula de Haversine
def calcular_distancia(coords_1, coords_2):
    if coords_1 and coords_2:
        distancia = geodesic(coords_1, coords_2).meters
        return distancia
    else:
        return None

# Função para criar o grafo do TSP
def criar_grafo_tsp(pedidos_df):
    G = nx.Graph()
    enderecos = pedidos_df['Endereço Completo'].unique()
    
    # Adicionar o endereço de partida
    G.add_node(endereco_partida, pos=endereco_partida_coords)
    
    for endereco in enderecos:
        coords = (pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Latitude'].values[0],
                  pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Longitude'].values[0])
        G.add_node(endereco, pos=coords)
    
    for (endereco1, endereco2) in permutations([endereco_partida] + list(enderecos), 2):
        coords_1 = G.nodes[endereco1]['pos']
        coords_2 = G.nodes[endereco2]['pos']
        distancia = calcular_distancia(coords_1, coords_2)
        if distancia is not None:
            G.add_edge(endereco1, endereco2, weight=distancia)
    
    return G

# Função para resolver o TSP usando Algoritmo Genético
def resolver_tsp_genetico(G):
    def fitness(route):
        return sum(G.edges[route[i], route[i+1]]['weight'] for i in range(len(route) - 1)) + G.edges[route[-1], route[0]]['weight']

    def mutate(route):
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route

    def crossover(route1, route2):
        size = len(route1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = route1[start:end]
        pointer = 0
        for i in range(size):
            if route2[i] not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = route2[i]
        return child

    def genetic_algorithm(population, generations=1000, mutation_rate=0.01):
        for _ in range(generations):
            population = sorted(population, key=lambda route: fitness(route))
            next_generation = population[:2]
            for _ in range(len(population) // 2 - 1):
                parents = random.sample(population[:10], 2)
                child = crossover(parents[0], parents[1])
                if random.random() < mutation_rate:
                    child = mutate(child)
                next_generation.append(child)
            population = next_generation
        return population[0], fitness(population[0])

    nodes = list(G.nodes)
    population = [random.sample(nodes, len(nodes)) for _ in range(100)]
    best_route, best_distance = genetic_algorithm(population)
    return best_route, best_distance

# Resolver o TSP usando o algoritmo de aproximação
def resolver_tsp_aproximacao():
    G = nx.complete_graph(5)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)

    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True)
    print("Caminho TSP:", tsp_path)

# Função para resolver o TSP usando OR-Tools
def resolver_tsp_ortools(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, None

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    return route, solution.ObjectiveValue()

# Função para resolver o VRP usando OR-Tools
def resolver_vrp(dist_matrix, num_vehicles, depot):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        routes.append(route)

    return routes

# Função para otimizar o aproveitamento da frota usando programação linear
def otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters):
    pedidos_df['Nº Carga'] = 0
    pedidos_df['Placa'] = ""
    pedidos_df['Capac. Kg'] = None
    carga_numero = 1
    
    # Ajustar a capacidade da frota
    caminhoes_df['Capac. Kg'] *= (percentual_frota / 100)
    caminhoes_df['Capac. Cx'] *= (percentual_frota / 100)
    
    # Filtrar caminhões disponíveis
    caminhoes_df = caminhoes_df[caminhoes_df['Disponível'] == 'Ativo']
    
    # Agrupar pedidos por região
    pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters)
    
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao]
        
        for _, caminhao in caminhoes_df.iterrows():
            capacidade_peso = caminhao['Capac. Kg']
            capacidade_caixas = caminhao['Capac. Cx']
            
            pedidos_alocados = pedidos_regiao[(pedidos_regiao['Peso dos Itens'] <= capacidade_peso) & (pedidos_regiao['Qtde. dos Itens'] <= capacidade_caixas)]
            pedidos_alocados = pedidos_alocados.sample(n=min(max_pedidos, len(pedidos_alocados)))
            
            if not pedidos_alocados.empty:
                pedidos_df.loc[pedidos_alocados.index, 'Nº Carga'] = carga_numero
                pedidos_df.loc[pedidos_alocados.index, 'Placa'] = caminhao['Placa']
                pedidos_df.loc[pedidos_alocados.index, 'Capac. Kg'] = caminhao['Capac. Kg']
                
                capacidade_peso -= pedidos_alocados['Peso dos Itens'].sum()
                capacidade_caixas -= pedidos_alocados['Qtde. dos Itens'].sum()
                
                carga_numero += 1
    
    # Verificar se as placas e números de carga foram atribuídos corretamente
    if pedidos_df['Placa'].isnull().any() or pedidos_df['Nº Carga'].isnull().any():
        st.error("Não foi possível atribuir placas ou números de carga a alguns pedidos. Verifique os dados e tente novamente.")
    
    # Ordena os pedidos mantendo o Nº Pedido original
    pedidos_df = pedidos_df.sort_values(by=['Nº Carga', 'Nº Pedido'])

    # Verifica se 'Nº Carga' possui valores nulos; se sim, alerta o usuário
    if pedidos_df['Nº Carga'].isnull().any():
        st.error("Existem pedidos sem número de carga definido. Verifique os dados.")
    else:
        # Cria a coluna "Ordem de Entrega" para cada grupo de "Nº Carga"
        pedidos_df['Ordem de Entrega'] = pedidos_df.groupby('Nº Carga').cumcount() + 1
        pedidos_df['Ordem de Entrega'] = pedidos_df['Ordem de Entrega'].astype(str) + " Entrega"

    return pedidos_df

# Função para cadastrar caminhões
def cadastrar_caminhoes():
    st.title("Cadastro de Caminhões da Frota")
    
    # Carregar DataFrame existente ou criar um novo
    try:
        caminhoes_df = pd.read_excel("caminhoes_frota.xlsx", engine='openpyxl')
    except FileNotFoundError:
        caminhoes_df = pd.DataFrame(columns=['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível'])
    
    # Upload do arquivo Excel de Caminhões
    uploaded_caminhoes = st.file_uploader("Escolha o arquivo Excel de Caminhões", type=["xlsx", "xlsm"])
    
    if uploaded_caminhoes is not None:
        # Salva o arquivo no database
        file_content = uploaded_caminhoes.getvalue()
        save_upload(DB_USERNAME, uploaded_caminhoes.name, file_content)
        st.success("Arquivo de caminhões salvo no database com sucesso!")
        
        novo_caminhoes_df = pd.read_excel(uploaded_caminhoes, engine='openpyxl')
        
        # Verificar se as colunas necessárias estão presentes
        colunas_caminhoes = ['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível']
        
        if not all(col in novo_caminhoes_df.columns for col in colunas_caminhoes):
            st.error("As colunas necessárias não foram encontradas na planilha de caminhões.")
            return
        
        # Excluir placas específicas
        placas_excluir = ["FLB1111", "FLB2222", "FLB3333", "FLB4444", "FLB5555", "FLB6666", "FLB7777", "FLB8888", "FLB9999"]
        novo_caminhoes_df = novo_caminhoes_df[~novo_caminhoes_df['Placa'].isin(placas_excluir)]
        
        # Botão para carregar a frota
        if st.button("Carregar Frota"):
            caminhoes_df = pd.concat([caminhoes_df, novo_caminhoes_df], ignore_index=True)
            caminhoes_df.to_excel("caminhoes_frota.xlsx", index=False)
            st.success("Frota carregada com sucesso!")

    # Botão para limpar a frota
    if st.button("Limpar Frota"):
        caminhoes_df = pd.DataFrame(columns=['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível'])
        caminhoes_df.to_excel("caminhoes_frota.xlsx", index=False)
        st.success("Frota limpa com sucesso!")
    
    # Exibir caminhões cadastrados
    st.subheader("Caminhões Cadastrados")
    edited_caminhoes_df = st.data_editor(caminhoes_df, num_rows="dynamic")
    
    # Botão para salvar alterações
    if st.button("Salvar Alterações"):
        edited_caminhoes_df.to_excel("caminhoes_frota.xlsx", index=False)
        st.success("Alterações salvas com sucesso!")

# Função para subir planilhas de roteirizações
def subir_roterizacoes():
    st.title("Upload de Planilhas de Roteirizações")
    
    # Carregar DataFrame existente ou criar um novo
    try:
        roterizacao_df = pd.read_excel("roterizacao_dados.xlsx", engine='openpyxl')
    except FileNotFoundError:
        roterizacao_df = pd.DataFrame(columns=['Placa', 'Nº Carga', 'Nº Pedido', 'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 'Qtde. dos Itens', 'Peso dos Itens'])
    
    # Upload do arquivo Excel de Roteirizações
    uploaded_roterizacao = st.file_uploader("Escolha o arquivo Excel de Roteirizações", type=["xlsx", "xlsm"])
    
    if uploaded_roterizacao is not None:
        # Salva o arquivo no database
        file_content = uploaded_roterizacao.getvalue()
        save_upload(DB_USERNAME, uploaded_roterizacao.name, file_content)
        st.success("Arquivo de roteirização salvo no database com sucesso!")
        
        novo_roterizacao_df = pd.read_excel(uploaded_roterizacao, engine='openpyxl')
        
        # Verificar se as colunas necessárias estão presentes
        colunas_roterizacao = ['Placa', 'Nº Carga', 'Nº Pedido', 'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 'Qtde. dos Itens', 'Peso dos Itens']
        
        colunas_faltando = [col for col in colunas_roterizacao if col not in novo_roterizacao_df.columns]
        if colunas_faltando:
            st.error(f"As seguintes colunas estão faltando na planilha de roteirizações: {', '.join(colunas_faltando)}")
            return
        
        # Botão para carregar a roteirização
        if st.button("Carregar Roteirização"):
            roterizacao_df = pd.concat([roterizacao_df, novo_roterizacao_df], ignore_index=True)
            roterizacao_df.to_excel("roterizacao_dados.xlsx", index=False)
            st.success("Roteirização carregada com sucesso!")
    
    # Botão para limpar a roteirização
    if st.button("Limpar Roteirização"):
        roterizacao_df = pd.DataFrame(columns=colunas_roterizacao)
        roterizacao_df.to_excel("roterizacao_dados.xlsx", index=False)
        st.success("Roteirização limpa com sucesso!")
    
    # Exibir dados da planilha de roteirizações
    st.subheader("Dados da Roteirização")
    st.dataframe(roterizacao_df)

# Função para aplicar estilos Material Design
def material_css():
    st.markdown("""
    <style>
    /* Importa a fonte Roboto */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Define a fonte geral */
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    /* Botões customizados */
    .stButton button {
        background-color: #6200EE; /* Roxo Material */
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3700B3;
    }

    /* Inputs customizados */
    .stTextInput > div > div > input {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 0.5rem;
    }

    /* Cabeçalho customizado */
    .css-18e3th9 {
        background-color: #6200EE;
        color: white;
        padding: 1rem;
    }

    /* Sidebar customizada */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# Função principal para o painel interativo
def main():
    material_css()  # Aplica os estilos Material Design
    st.title("Roteirizador de Pedidos")

    # Formulário de login
    st.sidebar.subheader("Login no Database")
    username_input = st.sidebar.text_input("Usuário")
    password_input = st.sidebar.text_input("Senha", type="password")
    login_button = st.sidebar.button("Entrar")

    db_autenticado = False
    if login_button:
        if username_input == DB_USERNAME and password_input == DB_PASSWORD:
            st.sidebar.success("Autenticado com sucesso!")
            db_autenticado = True
        else:
            st.sidebar.error("Usuário ou senha incorretos.")

    # Upload do arquivo Excel de Pedidos
    uploaded_pedidos = st.file_uploader("Escolha o arquivo Excel de Pedidos", type=["xlsx", "xlsm"])
    
    if uploaded_pedidos is not None:
        # Se estiver autenticado, salva o arquivo no DB
        if db_autenticado:
            file_content = uploaded_pedidos.getvalue()
            save_upload(username_input, uploaded_pedidos.name, file_content)
            st.success("Arquivo salvo no database com sucesso!")
        
        # Leitura das planilhas
        pedidos_df = pd.read_excel(uploaded_pedidos, engine='openpyxl')
        
        # Formar o endereço completo
        pedidos_df['Endereço Completo'] = pedidos_df['Endereço de Entrega'] + ', ' + pedidos_df['Bairro de Entrega'] + ', ' + pedidos_df['Cidade de Entrega']
        
        # Obter coordenadas geográficas sem precisar chamar a API se já estiver no banco
        with st.spinner('Aguarde, obtendo coordenadas de latitude e longitude...'):
            pedidos_df['Latitude'] = pedidos_df['Endereço Completo'].apply(lambda x: obter_coordenadas_com_fallback(x)[0])
            pedidos_df['Longitude'] = pedidos_df['Endereço Completo'].apply(lambda x: obter_coordenadas_com_fallback(x)[1])
        
        # Verificar se as coordenadas foram obtidas corretamente
        if pedidos_df['Latitude'].isnull().any() or pedidos_df['Longitude'].isnull().any():
            st.error("Não foi possível obter as coordenadas para alguns endereços. Verifique os endereços e tente novamente.")
            return
        
        # Carregar dados da frota cadastrada
        try:
            caminhoes_df = pd.read_excel("caminhoes_frota.xlsx", engine='openpyxl')
        except FileNotFoundError:
            st.error("Nenhum caminhão cadastrado. Por favor, cadastre caminhões primeiro.")
            return
        
        # Verificar se as colunas necessárias estão presentes
        colunas_pedidos = ['Nº Carga', 'Nº Pedido', 'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 'Qtde. dos Itens', 'Peso dos Itens']
        
        colunas_faltando_pedidos = [col for col in colunas_pedidos if col not in pedidos_df.columns]
        if colunas_faltando_pedidos:
            st.error(f"As seguintes colunas estão faltando na planilha de pedidos: {', '.join(colunas_faltando_pedidos)}")
            return
        
        colunas_caminhoes = ['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível']
        colunas_faltando_caminhoes = [col for col in colunas_caminhoes if col not in caminhoes_df.columns]
        if colunas_faltando_caminhoes:
            st.error(f"As seguintes colunas estão faltando na planilha da frota: {', '.join(colunas_faltando_caminhoes)}")
            return
        
        # Filtrar caminhões ativos
        caminhoes_df = caminhoes_df[caminhoes_df['Disponível'] == 'Ativo']
        
        # Opções de configuração
        n_clusters = st.slider("Número de regiões para agrupar", min_value=1, max_value=10, value=5)
        percentual_frota = st.slider("Capacidade da frota a ser usada (%)", min_value=0, max_value=100, value=100)
        max_pedidos = st.slider("Número máximo de pedidos por veículo", min_value=1, max_value=20, value=10)
        rota_tsp = st.checkbox("Aplicar TSP")
        rota_vrp = st.checkbox("Aplicar VRP")
        
        # Mostrar opções de roteirização após o upload da planilha
        if st.button("Roteirizar"):
            # Processamento dos dados
            pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
            
            # Agrupar por região
            pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters)
            
            # Alocar pedidos nos caminhões respeitando os limites de peso e quantidade de caixas
            pedidos_df = otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters)
            
            if rota_tsp:
                G = criar_grafo_tsp(pedidos_df)
                melhor_rota, menor_distancia = resolver_tsp_genetico(G)
                st.write("Melhor rota TSP:")
                st.write("\n".join(melhor_rota))
                st.write(f"Menor distância TSP: {menor_distancia}")
                pedidos_df['Ordem de Entrega TSP'] = pedidos_df['Endereço Completo'].apply(lambda x: melhor_rota.index(x) + 1)
            
            if rota_vrp:
                # Calcular a matriz de distâncias
                dist_matrix = calcular_matriz_distancias(pedidos_df)

                # Número de veículos (baseado no número de caminhões disponíveis)
                num_vehicles = len(caminhoes_df)

                # Índice do depósito (endereço de partida)
                depot = 0  # Geralmente o primeiro ponto na matriz de distâncias

                # Resolver o VRP
                melhor_rota_vrp = resolver_vrp(dist_matrix, num_vehicles, depot)
                st.write(f"Melhor rota VRP: {melhor_rota_vrp}")
            
            # Ordena os pedidos mantendo o Nº Pedido original
            pedidos_df = pedidos_df.sort_values(by=['Nº Carga', 'Nº Pedido'])

            # Cria a coluna "Ordem de Entrega" para cada grupo de "Nº Carga"
            # A contagem reinicia para cada novo grupo (carga)
            pedidos_df['Ordem de Entrega'] = pedidos_df.groupby('Nº Carga').cumcount() + 1
            pedidos_df['Ordem de Entrega'] = pedidos_df['Ordem de Entrega'].astype(str) + " Entrega"

            # Exibe a tabela com os dados originais e a nova coluna
            st.write("Dados dos pedidos:")
            st.dataframe(pedidos_df[['Placa', 'Nº Carga', 'Nº Pedido', 'Ordem de Entrega', 
                                     'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 
                                     'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 
                                     'Qtde. dos Itens', 'Peso dos Itens']])
            
            st.write("Dados dos pedidos com Ordem de Entrega e Coordenadas:")
            st.dataframe(pedidos_df[['Nº Carga', 'Nº Pedido', 'Ordem de Entrega',
                                     'Placa', 'Capac. Kg',
                                     'Latitude', 'Longitude',
                                     'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 
                                     'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 
                                     'Qtde. dos Itens', 'Peso dos Itens']])
            
            # Criar e exibir mapa
            mapa = criar_mapa(pedidos_df)
            folium_static(mapa)
            
            # Gerar arquivo Excel com a roteirização feita
            output_file_path = 'roterizacao_resultado.xlsx'
            pedidos_df.to_excel(output_file_path, index=False)
            st.write(f"Arquivo Excel com a roteirização feita foi salvo em: {output_file_path}")
            
            # Botão para baixar o arquivo Excel
            with open(output_file_path, "rb") as file:
                btn = st.download_button(
                    label="Baixar planilha",
                    data=file,
                    file_name="roterizacao_resultado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Criar um GeoDataFrame com coordenadas
            data = {
                'Endereço': pedidos_df['Endereço Completo'],
                'Latitude': pedidos_df['Latitude'],
                'Longitude': pedidos_df['Longitude']
            }
            gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Longitude'], data['Latitude']))
            
            # Salvar como arquivo GeoJSON
            gdf.to_file("rotas.geojson", driver="GeoJSON")
            st.write("Arquivo GeoJSON com as rotas foi salvo como 'rotas.geojson'")
    
    # Opção para cadastrar caminhões
    if st.checkbox("Cadastrar Caminhões"):
        cadastrar_caminhoes()
    
    # Opção para subir planilhas de roteirizações
    if st.checkbox("Subir Planilhas de Roteirizações"):
        subir_roterizacoes()

# Função para agrupar pedidos por região usando KMeans
def agrupar_por_regiao_kmeans(pedidos_df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    pedidos_df['Regiao'] = kmeans.fit_predict(pedidos_df[['Latitude', 'Longitude']])
    return pedidos_df

# Função para agrupar pedidos por região usando DBSCAN
def agrupar_por_regiao(pedidos_df, eps=0.01, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    # Converte latitude e longitude para radianos (necessário para DBSCAN com métrica haversine)
    coords = pedidos_df[['Latitude', 'Longitude']].apply(lambda x: np.radians(x))
    pedidos_df['Regiao'] = dbscan.fit_predict(coords)
    return pedidos_df

# Função para calcular a matriz de distâncias
def calcular_matriz_distancias(pedidos_df):
    enderecos = pedidos_df[['Latitude', 'Longitude']].values
    n = len(enderecos)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = geodesic(enderecos[i], enderecos[j]).meters
    return dist_matrix

# Função para criar o mapa com as rotas
def criar_mapa(pedidos_df):
    mapa = folium.Map(location=endereco_partida_coords, zoom_start=12)
    
    # Adicionar marcadores para os pedidos
    for _, row in pedidos_df.iterrows():
        popup_text = f"<b>Placa: {row['Placa']}</b><br>Endereço: {row['Endereço Completo']}"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            icon=folium.Icon(color='blue')
        ).add_to(mapa)
    
    # Adicionar marcador para o endereço de partida
    folium.Marker(
        location=endereco_partida_coords,
        popup="Endereço de Partida",
        icon=folium.Icon(color='red')
    ).add_to(mapa)
    
    # Adicionar uma rota
    rota = [[-23.0838, -47.1336], [-23.1, -47.2], [-23.2, -47.3]]
    folium.PolyLine(rota, color="blue", weight=2.5, opacity=1).add_to(mapa)
    
    return mapa

main()
