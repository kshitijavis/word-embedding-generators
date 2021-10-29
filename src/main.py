import word_extractor
from embeddings_generator import EmbeddingsGenerator

def main():
    text_file = open("../setup_data/harrpotter_sorcerersstone.txt")
    words = word_extractor.extract_words(text_file)

    generator = EmbeddingsGenerator()
    embeddings = generator.generate_embeddings(words, 1)

    harry_emb = embeddings.get_embeddings_for_word("Harry")

    print(harry_emb)

    text_file.close()


if __name__ == "__main__":
    main()