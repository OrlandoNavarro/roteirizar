import requests
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import networkx as nx
from itertools import permutations
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import random

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
def obter_coordenadas_com_fallback(endereco, coordenadas_salvas):
    if endereco in coordenadas_salvas:
        return coordenadas_salvas[endereco]
    
    coords = obter_coordenadas_opencage(endereco)
    if coords is None:
        # Coordenadas manuais para endereços específicos
        coordenadas_manuais = {
            "Rua Araújo Leite, 146, Centro, Piedade, São Paulo, Brasil": (-23.71241093449893, -47.41796911054548)
        }
        coords = coordenadas_manuais.get(endereco, (None, None))
    
    if coords:
        coordenadas_salvas[endereco] = coords
    
    return coords

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

# Função para resolver o VRP usando OR-Tools
def resolver_vrp(pedidos_df, caminhoes_df):
    # Implementação do VRP usando OR-Tools
    pass

# Função para otimizar o aproveitamento da frota usando programação linear
def otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters):
    pedidos_df['Nº Carga'] = 0
    pedidos_df['Placa'] = ""
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
                
                capacidade_peso -= pedidos_alocados['Peso dos Itens'].sum()
                capacidade_caixas -= pedidos_alocados['Qtde. dos Itens'].sum()
                
                carga_numero += 1
    
    # Verificar se as placas e números de carga foram atribuídos corretamente
    if pedidos_df['Placa'].isnull().any() or pedidos_df['Nº Carga'].isnull().any():
        st.error("Não foi possível atribuir placas ou números de carga a alguns pedidos. Verifique os dados e tente novamente.")
    
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

# Função principal para o painel interativo
def main():
    st.title("Roteirizador de Pedidos")
    
    # Upload do arquivo Excel de Pedidos
    uploaded_pedidos = st.file_uploader("Escolha o arquivo Excel de Pedidos", type=["xlsx", "xlsm"])
    
    if uploaded_pedidos is not None:
        # Leitura das planilhas
        pedidos_df = pd.read_excel(uploaded_pedidos, engine='openpyxl')
        
        # Formar o endereço completo
        pedidos_df['Endereço Completo'] = pedidos_df['Endereço de Entrega'] + ', ' + pedidos_df['Bairro de Entrega'] + ', ' + pedidos_df['Cidade de Entrega']
        
        # Carregar coordenadas salvas
        try:
            coordenadas_salvas_df = pd.read_excel("coordenadas_salvas.xlsx", engine='openpyxl')
            coordenadas_salvas = dict(zip(coordenadas_salvas_df['Endereço'], zip(coordenadas_salvas_df['Latitude'], coordenadas_salvas_df['Longitude'])))
        except FileNotFoundError:
            coordenadas_salvas = {}
        
        # Obter coordenadas geográficas
        with st.spinner('Aguarde, obtendo coordenadas de latitude e longitude...'):
            pedidos_df['Latitude'] = pedidos_df['Endereço Completo'].apply(lambda x: obter_coordenadas_com_fallback(x, coordenadas_salvas)[0])
            pedidos_df['Longitude'] = pedidos_df['Endereço Completo'].apply(lambda x: obter_coordenadas_com_fallback(x, coordenadas_salvas)[1])
        
        # Salvar coordenadas atualizadas
        coordenadas_salvas_df = pd.DataFrame(coordenadas_salvas.items(), columns=['Endereço', 'Coordenadas'])
        coordenadas_salvas_df[['Latitude', 'Longitude']] = pd.DataFrame(coordenadas_salvas_df['Coordenadas'].tolist(), index=coordenadas_salvas_df.index)
        coordenadas_salvas_df.drop(columns=['Coordenadas'], inplace=True)
        coordenadas_salvas_df.to_excel("coordenadas_salvas.xlsx", index=False)
        
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
                melhor_rota_vrp = resolver_vrp(pedidos_df, caminhoes_df)
                st.write(f"Melhor rota VRP: {melhor_rota_vrp}")
            
            # Exibir resultado
            st.write("Dados dos pedidos:")
            st.dataframe(pedidos_df[['Placa', 'Nº Carga', 'Nº Pedido', 'Cód. Cliente', 'Nome Cliente', 'Grupo Cliente', 'Endereço de Entrega', 'Bairro de Entrega', 'Cidade de Entrega', 'Qtde. dos Itens', 'Peso dos Itens']])
            
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
    
    # Opção para cadastrar caminhões
    if st.checkbox("Cadastrar Caminhões"):
        cadastrar_caminhoes()
    
    # Opção para subir planilhas de roteirizações
    if st.checkbox("Subir Planilhas de Roteirizações"):
        subir_roterizacoes()

# Função para agrupar pedidos por região usando KMeans
def agrupar_por_regiao(pedidos_df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    pedidos_df['Regiao'] = kmeans.fit_predict(pedidos_df[['Latitude', 'Longitude']])
    return pedidos_df

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
    
    return mapa

main()
