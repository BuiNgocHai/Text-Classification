import torch
from importlib import import_module
import pandas as pd
import tqdm as tqdm
import re

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
model = x.Model(config).to('cpu')
model.load_state_dict(torch.load(config.save_path, map_location='cuda'))


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


if __name__ == '__main__':
    print(predict("When Baby John wants the attention of his mommy and daddy, he is taught the important lesson of good manners.  It is important that Baby John be patient and respectful of others.  Enjoy this new song Wait Your Turn by Little Angel  #littleangel  #littleangelnurseryrhymes  #babyjohnsongs"))
    # data = pd.read_csv("./social_test.csv")
    # data['class'] = ''
    # for row in range(len(data)):
    #     text = data['text'][row]
    #     text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)
    #     text = text.replace('\t', '').replace('\n' ,'')
    #     data['class'][row] = predict(text)
    #     # break
    # data.to_csv('predict.csv', encoding='utf-8')
