import numpy as np
from .embeddings import WordEmbeddings

class CountGenerator:
    def __init__(self, window_size: int):
        self.window_size: int = window_size

    def generate_embeddings(self, words):
        word_embedding_inds = self.__get_word_inds(words)
        unique_word_count = len(word_embedding_inds)
        embeddings_matrix = np.zeros((unique_word_count, unique_word_count))

        for focus_word_ind, focus_word in enumerate(words):
            focus_embedding_ind = word_embedding_inds[focus_word]

            window_start = max(0, focus_word_ind - self.window_size)
            window_end = min(len(words) - 1, focus_word_ind + self.window_size) # inclusive

            for context_word in words[window_start:(window_end + 1)]:
                context_embedding_ind = word_embedding_inds[context_word]

                embeddings_matrix[focus_embedding_ind][context_embedding_ind] += 1

        return WordEmbeddings(embeddings_matrix, word_embedding_inds)

    def __get_word_inds(self, words) -> dict:
        word_set = set(words)
        return {word:ind for ind, word in enumerate(word_set)}
