import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import matplotlib.pyplot as plt


def visualize_graph(G):
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, node_color="green", font_size=4, font_weight="bold",
            edge_color="gray")
    plt.show()


def _create_graph(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for node1, node2, weight in edges:
        # print(f"Adding edge between ({node1}) and ({node2}) with weight {weight}")
        G.add_edge(node1, node2, weight=weight)
    return G


def _remove_most_weighted_nodes(G):
    while G.number_of_edges() > 0:
        weight_sums = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in
                       G.nodes()}
        sorted_nodes = sorted(weight_sums, key=lambda x: weight_sums[x], reverse=True)
        node_to_remove = sorted_nodes[0]
        G.remove_node(node_to_remove)


class SimilarityGraph:
    def __init__(self, vectorizer_class: callable, threshold=0.8):
        self.threshold = threshold
        self.vectorizer = vectorizer_class

    def get_tfidf_cosine_matrix(self, data: list):
        vectorizer = self.vectorizer().fit_transform(data)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)
        assert numpy.array_equal(cosine_matrix, cosine_matrix.T)
        return cosine_matrix

    def _detect_similar_entries(self, data: dict):
        data_keys = list(data.keys())
        cosine_matrix = self.get_tfidf_cosine_matrix(list(data.values()))
        similar_pairs = []
        for i in range(len(data_keys)):
            for j in range(i + 1, len(data_keys)):
                if cosine_matrix[i][j] > self.threshold:
                    similar_pairs.append((data_keys[i], data_keys[j], cosine_matrix[i][j]))

        return similar_pairs

    def detect_and_remove_similar_entries(self, data: dict):
        similar_pairs = self._detect_similar_entries(data)
        G = _create_graph(list(data.keys()), similar_pairs)
        _remove_most_weighted_nodes(G)
        return G
