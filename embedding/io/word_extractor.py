from io import TextIOWrapper


def extract_words(file: TextIOWrapper):
    words = []
    for line in file:
        for word in line.split():
            words.append(word)

    return words