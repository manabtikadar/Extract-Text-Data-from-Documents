import spacy
from spacy.tokens import DocBin
import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_spacy_data(data, file_path, nlp):
    db = DocBin()
    for text, annotations in data:
        doc = nlp(text)
        doc.ents = [doc.char_span(start, end, label=label) for start, end, label in annotations['entities']]
        db.add(doc)
    db.to_disk(file_path)
    print(f"Saved: {file_path}")

nlp = spacy.blank("en")

train_data = load_pickle(r"C:\Users\manab\OneDrive\Desktop\Project\pickelfile\trainData.pickle")
test_data = load_pickle(r"C:\Users\manab\OneDrive\Desktop\Project\pickelfile\TestData.pickle")

save_spacy_data(train_data, "./train.spacy", nlp)
save_spacy_data(test_data, "./test.spacy", nlp)