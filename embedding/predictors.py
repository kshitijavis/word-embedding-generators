import numpy as np
from .embeddings import WordEmbeddings


class NextWordPredictor:
    def __init__(self, embeddings: WordEmbeddings):
        self.embeddings = embeddings

    def get_other_word_likelihoods(self, word: str) -> dict:
        word_embs = self.embeddings.get_embeddings_for_word(word)
        counts = {}

        for word, ind in self.embeddings.word_inds.items():
            count = word_embs[ind]
            if count != 0:
                counts[word] = word_embs[ind]

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    def get_most_likely(self, word: str):
        word_embs = self.embeddings.get_embeddings_for_word(word)
        this_word_ind = self.embeddings.word_inds[word]

        # Skip input word when searching for most likely word
        mask = np.zeros(word_embs.shape)
        mask[this_word_ind] = True
        masked_word_embs = np.ma.array(word_embs, mask=mask)

        most_likely_ind = np.argmax(masked_word_embs)
        likelihood = word_embs[most_likely_ind]

        if likelihood == 0:
            return None
        
        for word, ind in self.embeddings.word_inds.items():
            if ind == most_likely_ind:
                return word
            
        return None