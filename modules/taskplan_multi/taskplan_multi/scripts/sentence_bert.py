from sentence_transformers import SentenceTransformer


def save_model():
    modelPath = "./data/sentence_transformers/"
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    model.save(modelPath)


if __name__ == "__main__":
    # Parse the command line args and set device
    save_model()
