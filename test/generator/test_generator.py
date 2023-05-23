import unittest

from embedding.generator.generator import EmbeddingsGenerator

class TestGenerator(unittest.TestCase):
    def test_window_size_1(self):
        sentence = "a b c"
        words = sentence.split(" ")

        generator = EmbeddingsGenerator()
        embeddings = generator.generate_embeddings(words, 1)

        word_inds = embeddings.word_inds

        a_embs = embeddings.get_embeddings_for_word("a")
        self.assertEqual(a_embs[word_inds["a"]], 1)
        self.assertEqual(a_embs[word_inds["b"]], 1)
        self.assertEqual(a_embs[word_inds["c"]], 0)

        b_embs = embeddings.get_embeddings_for_word("b")
        self.assertEqual(b_embs[word_inds["a"]], 1)
        self.assertEqual(b_embs[word_inds["b"]], 1)
        self.assertEqual(b_embs[word_inds["c"]], 1)

        c_embs = embeddings.get_embeddings_for_word("c")
        self.assertEqual(c_embs[word_inds["a"]], 0)
        self.assertEqual(c_embs[word_inds["b"]], 1)
        self.assertEqual(c_embs[word_inds["c"]], 1)

    def test_window_size_1_multiple_occurrence(self):
        sentence = "a b c a b b a"
        words = sentence.split(" ")

        generator = EmbeddingsGenerator()
        embeddings = generator.generate_embeddings(words, 1)

        word_inds = embeddings.word_inds

        a_embs = embeddings.get_embeddings_for_word("a")
        self.assertEqual(a_embs[word_inds["a"]], 3)
        self.assertEqual(a_embs[word_inds["b"]], 3)
        self.assertEqual(a_embs[word_inds["c"]], 1)

        b_embs = embeddings.get_embeddings_for_word("b")
        self.assertEqual(b_embs[word_inds["a"]], 3)
        self.assertEqual(b_embs[word_inds["b"]], 5)
        self.assertEqual(b_embs[word_inds["c"]], 1)

        c_embs = embeddings.get_embeddings_for_word("c")
        self.assertEqual(c_embs[word_inds["a"]], 1)
        self.assertEqual(c_embs[word_inds["b"]], 1)
        self.assertEqual(c_embs[word_inds["c"]], 1) 

    def test_window_size_2_multiple_occurrence(self):
        sentence = "a b c a b b a"
        words = sentence.split(" ")

        generator = EmbeddingsGenerator()
        embeddings = generator.generate_embeddings(words, 2)

        word_inds = embeddings.word_inds

        a_embs = embeddings.get_embeddings_for_word("a")
        self.assertEqual(a_embs[word_inds["a"]], 3)
        self.assertEqual(a_embs[word_inds["b"]], 6)
        self.assertEqual(a_embs[word_inds["c"]], 2)

        b_embs = embeddings.get_embeddings_for_word("b")
        self.assertEqual(b_embs[word_inds["a"]], 6)
        self.assertEqual(b_embs[word_inds["b"]], 5)
        self.assertEqual(b_embs[word_inds["c"]], 2)

        c_embs = embeddings.get_embeddings_for_word("c")
        self.assertEqual(c_embs[word_inds["a"]], 2)
        self.assertEqual(c_embs[word_inds["b"]], 2)
        self.assertEqual(c_embs[word_inds["c"]], 1)    