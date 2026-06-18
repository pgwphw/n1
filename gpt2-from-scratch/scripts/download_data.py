from gpt2_scratch.data import download_tiny_shakespeare


if __name__ == "__main__":
    path = download_tiny_shakespeare("data/input.txt")
    print(path)
