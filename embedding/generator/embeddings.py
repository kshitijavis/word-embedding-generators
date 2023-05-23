import numpy as np

class WordEmbeddings:
    def __init__(self, embeddings_matrix: np.ndarray, word_inds: dict):
        word_count = len(word_inds)

        expected_shape = (word_count, word_count)
        if embeddings_matrix.shape != expected_shape:
            raise ValueError(f"Shape of embeddings matrix should be {expected_shape}")

        self.word_inds = word_inds
        self.embeddings_matrix = embeddings_matrix

    def get_embeddings_for_word(self, word):
        if word not in self.word_inds:
            raise KeyError("Word wasn't found in embeddings matrix")

        word_ind = self.word_inds[word]
        return self.embeddings_matrix[word_ind]