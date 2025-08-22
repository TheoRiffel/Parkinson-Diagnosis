from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import os

from tqdm import tqdm

class FD20Builder:
    """ FD20 Builder
    A classe constrói um grafo a partir de uma matriz de adjacência.
    O grafo gerado são a partir dos top 20% de arestas mais fortes."""

    def __init__(self, adj_matrix: np.ndarray):

        if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Input must be a square 2D NumPy array.")
        
        self.adj_matrix = adj_matrix
        self.final_graph: nx.Graph = nx.Graph()

    def build(self, verbose: bool = False) -> nx.Graph:
        np.fill_diagonal(self.adj_matrix, 0)
        threshold = np.percentile(self.adj_matrix, 80)
        self.adj_matrix[self.adj_matrix < threshold] = 0

        self.final_graph = nx.from_numpy_array(self.adj_matrix)
        return self.final_graph


class FDBuilder:
    """ FD Builder
    A classe constrói um grafo a partir de uma matriz de adjacência.
    O grafo gerado são a partir dos top N% de arestas mais fortes. Sendo N o valor passado como parâmetro."""

    def __init__(self, adj_matrix: np.ndarray, n_percent: float = 20.00):

        if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Input must be a square 2D NumPy array.")
        
        if not (0 < n_percent <= 100):
            raise ValueError("n_percent must be between 0 and 100.")
        
        self.adj_matrix = adj_matrix
        self.n_percent = n_percent
        self.final_graph: nx.Graph = nx.Graph()

    def build(self) -> nx.Graph:
        np.fill_diagonal(self.adj_matrix, 0)
        threshold = np.percentile(self.adj_matrix, 100 - self.n_percent)
        self.adj_matrix[self.adj_matrix < threshold] = 0

        self.final_graph = nx.from_numpy_array(self.adj_matrix)
        return self.final_graph

class OMSTBuilder:
    """ Orthogonal Minimal Spanning Tree (OMST) Builder
    A classe encontra iterativamente Orthogonal Minimal Spanning Trees (OMSTs)
    e as adicionam a um grafo cumulativo, enquanto a
    Eficiência do Custo Global (GCE) aumentar.
    """

    def __init__(self, adj_matrix: np.ndarray):

        if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Input must be a square 2D NumPy array.")

        # Correspondente ao input 'CIJ' no script MATLAB.
        # Utilizando abs() para garantir pesos positivos, típico das matrizes de correlação.
        self.adj_matrix = np.abs(adj_matrix)
        self.n_nodes = self.adj_matrix.shape[0]

        # --- Inicializa as métricas do grafo original ---
        # Correspondente à 'cost_ini'
        self._initial_cost = np.sum(np.triu(self.adj_matrix))
        # Correspondente à 'E_ini'
        self._initial_ge = self._calculate_global_efficiency_from_matrix(self.adj_matrix)

        # --- Initialize graphs for the build process ---
        # Correspondente à 'CIJnotintree' - grafo das arestas restantes.
        # Utilizando grafos NetworkX para eficiência.
        self._residual_graph = self._create_distance_graph(self.adj_matrix)

        # --- Atributos públicos para armazenar os resultados ---
        self.omsts: List[nx.Graph] = []
        self.gce_scores: List[float] = []
        self.final_graph: nx.Graph = nx.Graph()

    def _create_distance_graph(self, matrix: np.ndarray) -> nx.Graph:
        """
        Auxiliar para criar um grafo ponderado por distância a partir de uma matriz de similaridade.
        Nos grafos de distância, menor peso é melhor.
        Correspondente à `1./CIJ` no script MATLAB.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            distance_matrix = 1 / matrix
        # Definido valores não-finitos (1/0 ou NaNs) para 0, indicando nenhum caminho.
        distance_matrix[~np.isfinite(distance_matrix)] = 0
        return nx.from_numpy_array(distance_matrix)

    def _calculate_global_efficiency_from_matrix(self, matrix: np.ndarray) -> float:
        """Calcula a eficiência global da matriz de similaridade."""
        dist_graph = self._create_distance_graph(matrix)
        return nx.global_efficiency(dist_graph)

    def _calculate_gce(self, graph: nx.Graph) -> float:
        """
        Calcula a Eficiência de Custo Global para o grafo cumulativo atual.
        Correspondente à formula `E/E_ini - cost(counter)`.
        """
        if self._initial_cost == 0 or self._initial_ge == 0:
            return -np.inf

        # Custo é a soma dos pesos (similaridade) como uma fração do custo total inicial.
        current_cost = graph.size(weight='weight') / self._initial_cost

        # Eficiência calculada no ggrafo de distância correspondente.
        current_ge = self._calculate_global_efficiency_from_matrix(nx.to_numpy_array(graph))
        
        return (current_ge / self._initial_ge) - current_cost

    def build(self, verbose: bool = False) -> nx.Graph:
        """
        Executa o processo de construção iterativo, correspondente ao loop principal
        `while delta > 0` no script MATLAB.

        Args:
            verbose: Se verdadeiro, imprime a pontuação do GCE em cada iteração.

        Retorna:
            O Grafo construído com OMST.
        """
        # Este grafo acumulará as arestas dos OMSTs encontrados.
        cumulative_graph = nx.Graph()
        previous_gce = -np.inf

        if verbose:
            print(f"Starting build. Initial GE={self._initial_ge:.4f}, Initial Cost={self._initial_cost:.2f}")
            print("-" * 30)

        for i in range(self.n_nodes * (self.n_nodes - 1) // 2): # Max possible iterations
            if self._residual_graph.number_of_edges() == 0:
                if verbose: print("\nNo more edges available. Stopping.")
                break

            # Encontrando o próximo MST a partir das arestas restantes.
            mst = nx.minimum_spanning_tree(self._residual_graph, weight='weight')
            if mst.number_of_edges() == 0:
                if verbose: print("\nGraph disconnected. Stopping.")
                break
            
            candidate_graph = cumulative_graph.copy()
            for u, v in mst.edges():
                weight = self.adj_matrix[u, v]
                candidate_graph.add_edge(u, v, weight=weight)

            # Calculando o GCE deste novo grafo candidato.
            current_gce = self._calculate_gce(candidate_graph)

            if verbose:
                print(f"Iteration {i+1}: GCE = {current_gce:.4f}")

            # Condição de parada: se GCE não melhorar, grafo ótimo encontrado.
            if current_gce < previous_gce:
                if verbose:
                    print(f"GCE decreased. Halting at {len(self.omsts)} OMST(s).")
                    print("-" * 30)
                break

            previous_gce = current_gce
            self.gce_scores.append(current_gce)
            self.omsts.append(mst)
            cumulative_graph = candidate_graph
 
            self._residual_graph.remove_edges_from(mst.edges())
        
        self.final_graph = cumulative_graph
        return self.final_graph

    def plot_gce_curve(self):
        """Traçando a pontuação do GCE em relação ao número de OMSTs adicionados."""
        if not self.gce_scores:
            print("No scores to plot. Run .build() first.")
            return

        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.gce_scores) + 1)
        plt.plot(iterations, self.gce_scores, marker='o', linestyle='-')
        
        max_gce = max(self.gce_scores)
        max_idx = self.gce_scores.index(max_gce)
        plt.plot(max_idx + 1, max_gce, 'r*', markersize=15, label=f'Max GCE: {max_gce:.4f}')
        
        plt.title("Global Cost Efficiency vs. Number of OMSTs")
        plt.xlabel("Number of OMSTs Added")
        plt.ylabel("Global Cost Efficiency (GCE)")
        plt.grid(True)
        plt.legend()
        plt.show()

def concatenate_graphs(
        *graph_lists: list[nx.Graph],
    ) -> list[nx.Graph]:
    """
    Concatenate two or more list[nx.Graph]
    """

    if not graph_lists:
        raise ValueError("At least one list of networkx graphs must be provided for concatenation.")

    all_graphs = []
    for graph_list in graph_lists:
        all_graphs.extend(graph_list)

    return all_graphs

def get_largest_connected_component(graphs):
    largest_cc_graphs = []
    for graph in graphs:
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
        subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
        largest_cc_graphs.append(subgraph)
    return largest_cc_graphs

def extract_network_features(graph: nx.Graph) -> dict:
    """
    Extract network features from a graph
    """
    graph_copy = graph.copy()
    largest_cc = max(nx.connected_components(graph_copy), key=len)
    subgraph = graph.subgraph(largest_cc).copy()
    subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
    
    features = {}
    features['num_nodes'] = subgraph.number_of_nodes()
    features['num_edges'] = subgraph.number_of_edges()
    features['density'] = nx.density(subgraph)
    features['avg_clustering'] = nx.average_clustering(subgraph)
    features['transitivity'] = nx.transitivity(subgraph)
    features['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
    features['diameter'] = nx.diameter(subgraph)
    features['radius'] = nx.radius(subgraph)
    features['assortativity'] = nx.degree_assortativity_coefficient(subgraph)
    features['global_efficiency'] = nx.global_efficiency(subgraph)
    features['local_efficiency'] = nx.local_efficiency(subgraph)
    features['mean_degree'] = np.mean(np.array(list(dict(subgraph.degree()).values())))
    features['closeness_centrality'] = np.mean(list(nx.closeness_centrality(subgraph).values()))
    features['betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(subgraph).values()))
    features['eigenvector_centrality'] = np.mean(list(nx.eigenvector_centrality_numpy(subgraph).values())) 
    features['modularity'] = nx.algorithms.community.modularity(subgraph, nx.community.greedy_modularity_communities(subgraph))
    
    return features

def load_graphs(path: str) -> list[nx.Graph]:
    """
    Load graphs from path
    """
    graphs = []
    ordered_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
    for file in tqdm(ordered_files):
        if file.endswith('.pt'):
            graph_path = os.path.join(path, file)
            graph = nx.read_graphml(graph_path)
            graphs.append(graph)
    
    print(f'Loaded {len(graphs)} graphs from {path}')
    return graphs

def save_graphs(graphs, path: str):

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, graph in enumerate(tqdm(graphs)):
        graph_path = os.path.join(path, f'graph_{idx}.pt')
        nx.write_graphml(graph, graph_path)
    print(f'Saved {len(graphs)} graphs to {path}')

