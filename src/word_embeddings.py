class WordEmbeddings:
    def __init__(self, embeddings_matrix, word_inds):
        self.word_inds = word_inds
        self.embeddings_matrix = embeddings_matrix

    def get_embeddings_for_word(self, word):
        if word not in self.word_inds:
            raise ValueError("Word wasn't found in embeddings matrix")

        word_ind = self.word_inds[word]
        return self.embeddings_matrix[word_ind]