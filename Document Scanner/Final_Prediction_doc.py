import numpy as np
import pandas as pd
import pytesseract
import cv2
import spacy
import re
import string
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load the trained NER model
model_ner = spacy.load(r'C:\Users\manab\OneDrive\Desktop\Project\Train\output\model-best')


def clean_text(txt):
    """Clean text by removing unnecessary whitespace and punctuation."""
    punctuation = '!#$%&\'()*+:;<=>?[\\\\]^`{|}~'
    text = str(txt).strip().lower()
    text = text.translate(str.maketrans('', '', punctuation))
    return text


# Group tokens by label
class GroupGen:
    def __init__(self):
        self.id = 0
        self.text = ''

    def get_group(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id


def parse_text(text, label):
    """Parse text based on label types."""
    if label == 'PHONE':
        text = re.sub(r'\D', '', text)
    elif label == 'EMAIL':
        text = re.sub(r'[^A-Za-z0-9@_.\-]', '', text)
    elif label == 'WEB':
        text = re.sub(r'[^A-Za-z0-9:/.%#\-]', '', text)
    elif label in ('NAME', 'DES'):
        text = re.sub(r'[^a-z ]', '', text.lower()).title()
    elif label == 'ORG':
        text = re.sub(r'[^a-z0-9 ]', '', text.lower()).title()
    return text


grp_gen = GroupGen()

def get_predictions(image):
    """Get predictions and entity tagging on the image."""
    # Extract data using pytesseract
    tessdata = pytesseract.image_to_data(image)
    tesslist = [x.split('\t') for x in tessdata.split('\n')]

    # Ensure valid DataFrame structure
    try:
        df = pd.DataFrame(tesslist[1:], columns=tesslist[0])
        df.dropna(inplace=True)
        df['text'] = df['text'].apply(clean_text)
        df_clean = df.query('text != ""')
    except Exception as e:
        print("Error processing tessdata:", e)
        return image, {}

    content = " ".join(df_clean['text'].tolist())
    print("Extracted content:", content)

    doc = model_ner(content)

    # Convert doc into JSON
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    # Create DataFrame for tokens
    df_tokens = pd.DataFrame(doc_json['tokens'])
    df_tokens['tokens'] = df_tokens[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

    # Merge with entity labels
    entity_table = pd.DataFrame(doc_json['ents'])[['start', 'label']]
    df_tokens = pd.merge(df_tokens, entity_table, how='left', on='start').fillna('O')

    # Add start and end positions to df_clean
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

    # Merge df_clean with df_tokens
    dataframe_info = pd.merge(df_clean, df_tokens[['start', 'tokens', 'label']], how='inner', on='start')

    # Filter for labeled bounding boxes
    boundingbox_dataframe = dataframe_info.query("label != 'O'")
    boundingbox_dataframe['label'] = boundingbox_dataframe['label'].apply(lambda x: x[2:])
    boundingbox_dataframe['group'] = boundingbox_dataframe['label'].apply(grp_gen.get_group)

    # Calculate right and bottom of bounding boxes
    boundingbox_dataframe[['left', 'top', 'width', 'height']] = boundingbox_dataframe[['left', 'top', 'width', 'height']].astype(int)
    boundingbox_dataframe['right'] = boundingbox_dataframe['left'] + boundingbox_dataframe['width']
    boundingbox_dataframe['bottom'] = boundingbox_dataframe['top'] + boundingbox_dataframe['height']

    # Group by 'group' and aggregate
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'tokens', 'group']
    grouped_df = boundingbox_dataframe[col_group].groupby('group').agg({
        'left': 'min',
        'right': 'max',
        'top': 'min',
        'bottom': 'max',
        'label': lambda x: x.unique(),
        'tokens': lambda x: " ".join(x)
    })

    # Draw grouped bounding boxes on the image
    image_bb = image.copy()
    for l, r, t, b, label, token in grouped_df.values:
        cv2.rectangle(image_bb, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(image_bb, str(label[0]), (l, t - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    # Extract entities
    info_array = dataframe_info[['tokens', 'label']].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]
        text = parse_text(token, label_tag)

        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] += " " + text
                    else:
                        entities[label_tag][-1] += text
            previous = label_tag

    return image_bb, entities


# # Load Image
# image_path = r'C:\Users\manab\OneDrive\Desktop\Project\Selected\IMG-20240812-WA0004[1].jpeg'

# if os.path.exists(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (640, 360))
#     image_bb, entities = get_predictions(image)
#     print("Extracted Entities:", entities)
#     image_array = np.array(image_bb)
#     plt.imshow(image_array)
#     plt.axis('off')
#     plt.show()
# else:
#     print("Image path does not exist.")
