import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding.io import word_extractor
from embedding.generator.generator import EmbeddingsGenerator

def main():
    text_file = open("data/harrpotter_sorcerersstone.txt")
    words = word_extractor.extract_words(text_file)

    generator = EmbeddingsGenerator()
    embeddings = generator.generate_embeddings(words, 1)

    harry_emb = embeddings.get_embeddings_for_word("Harry")

    print(harry_emb)

    text_file.close()


if __name__ == "__main__":
    main()