from embedding.embeddings import WordEmbeddings
import numpy as np
import unittest

class TestWordEmbeddings(unittest.TestCase):
    def test_small_matrix(self):
        word_inds = {
            "a": 0,
            "b": 1,
            "c": 2,
        }

        embs = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ])

        word_embeddings = WordEmbeddings(embs, word_inds)

        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("a"), np.array([0, 1, 1]))
        
        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("b"), np.array([1, 0, 0]))
        
        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("c"), np.array([1, 0, 0]))
        
    def test_out_of_order(self):
        word_inds = {
            "b": 1,
            "c": 2,
            "a": 0,
        }

        embs = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ])

        word_embeddings = WordEmbeddings(embs, word_inds)

        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("a"), np.array([0, 1, 1]))
        
        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("b"), np.array([1, 0, 0]))
        
        np.testing.assert_array_equal(
            word_embeddings.get_embeddings_for_word("c"), np.array([1, 0, 0]))
        
    def test_mismatched_size(self):
        word_inds = {
            "a": 0,
            "b": 1,
            "c": 2,
        }

        embs = np.array([
            [0, 1, 1],
            [1, 0, 0],
        ])

        with self.assertRaises(ValueError):
            WordEmbeddings(embs, word_inds)
