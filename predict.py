import torch
from importlib import import_module
import pandas as pd
import tqdm as tqdm
import re
import argparse
import os

parser = argparse.ArgumentParser(description='Bert predict')
parser.add_argument('--text', type=str, required=False, help='Text to predict')
parser.add_argument('--csv', type=str, required=False, help='Path to csv file want to predict')
parser.add_argument('--device', type=str, default='cpu', help='cuda|cpu')
args = parser.parse_args()


key = {
    0: 'beauty',
    1: 'fitness',
    2: 'food',
    3: 'parenting',
    4: 'sports'
}

model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config('Social')
#model = x.Model(config).to(config.device)
device = args.device
model = x.Model(config).to(device)
model.load_state_dict(torch.load(config.save_path, map_location=device))


def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    ids = torch.LongTensor([token_ids])
    seq_len = torch.LongTensor([seq_len])
    mask = torch.LongTensor([mask])
    return ids, seq_len, mask


def predict(text):
    """
    :param text:
    :return:
    """
    data = build_predict_text(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]


def preprocess(text):
    text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)
    text = text.replace('\t', '').replace('\n' ,'')
    return text


def process_csv(csv_path):
    if not os.path.isfile(csv_path):
        print("File not found")
        return False
    data = pd.read_csv(csv_path)
    data['class'] = ''
    print('Process data.....')
    for row in range(len(data)):
        text = data['text'][row]
        text = preprocess(text)
        data['class'][row] = predict(text)
    path_write = csv_path[:-4] + '_predict.csv'
    data.to_csv(path_write, encoding='utf-8')
    print('Save results in: ', path_write)
    return True


if __name__ == '__main__':
    if args.text:
        text = preprocess(args.text)
        print(predict(text))
    if args.csv:
        process_csv(args.csv)


