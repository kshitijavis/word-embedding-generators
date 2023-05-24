import unittest

import embedding.generators as generators
import embedding.predictors as predictors

class TestNextWordPredictor(unittest.TestCase):
    def test_most_likely_word(self):
        # Uses same sentence as test_window_size_2_multiple_occurrence
        sentence = "a b c a b b a"
        words = sentence.split(" ")

        generator = generators.CountGenerator(2)
        embeddings = generator.generate_embeddings(words)

        predictor = predictors.NextWordPredictor(embeddings)

        self.assertEqual(predictor.get_most_likely("a"), "b")
        self.assertEqual(predictor.get_most_likely("b"), "a")
        self.assertIn(predictor.get_most_likely("c"), ("a", "b"))