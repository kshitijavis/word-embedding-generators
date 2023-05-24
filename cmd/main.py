import embedding.tokenizers as token
from embedding.generators import CountGenerator

def main():
    with open("data/harrpotter_sorcerersstone.txt", "r") as f:
        text = f.read()

    tokenizer = token.Whitespace(text)
    words = tokenizer.get_tokens()

    generator = CountGenerator(1)
    embeddings = generator.generate_embeddings(words)


if __name__ == "__main__":
    main()