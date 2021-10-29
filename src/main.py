import word_extractor

def main():
    text_file = open("../setup_data/harrpotter_sorcerersstone.txt")
    words = word_extractor.extract_words(text_file)
    print(words)

    text_file.close()


if __name__ == "__main__":
    main()