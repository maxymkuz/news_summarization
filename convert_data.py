import os
import pandas as pd
from tqdm import tqdm


def get_texts():
    texts = []
    prefix_len = len('data\\articles\\')
    for root, subFolders, files in os.walk('data\\articles'):
        if not files:
            continue

        for file in files:
            texts.append(root[prefix_len:] + '\\' + file)

    return texts


if __name__ == '__main__':
    texts = get_texts()

    data = {'text': [], 'summary': []}
    for text in tqdm(texts):
        with open('\\'.join(['data\\articles', text])) as f:
            data['text'].append(f.read())
        with open('\\'.join(['data\\summaries', text])) as f:
            data['summary'].append(f.read())

    df = pd.DataFrame(data=data)
    df.to_csv('extractive_data.csv', index=False)
