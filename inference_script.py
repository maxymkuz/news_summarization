import numpy
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config

from tqdm import tqdm
import gc


class Config:

    model_name = "google/t5-v1_1-base"
    tokenizer_name = "t5-base"
    checkpoint_name = "models/t5_v11_base-e2-l1_707.ckpt"

    src_max_tokens = 768
    tgt_max_tokens = 256

    data_frac = 0.01

    batch_size = 16
    num_workers = 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')


df_inf = pd.read_csv('cnn_dailymail/test.csv')

# sample [Config.data_frac]% of data
df_inf = df_inf.sample(n=int(Config.data_frac * len(df_inf)))

print(f'Len of df: {len(df_inf)}')


class CNNDailyMailDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.iloc[idx]  # id article highlights

        data.article = 'summarize: ' + data.article
        src_tokens = self.tokenizer(data.article, return_tensors='pt',
                                    padding='max_length', truncation=True,
                                    max_length=Config.src_max_tokens)

        tgt_tokens = self.tokenizer(data.highlights, return_tensors='pt',
                                    padding='max_length', truncation=True,
                                    max_length=Config.tgt_max_tokens)

        # by convention, labels for ignored tokens (so just padding) should be set to -100
        tgt_tokens.input_ids[tgt_tokens.input_ids == self.tokenizer.pad_token_id] = -100

        # flatten is i guess a thing with T5? don't need it with other models
        return (src_tokens.input_ids.flatten(), src_tokens.attention_mask.flatten(),
                tgt_tokens.input_ids.flatten(), tgt_tokens.attention_mask.flatten())


t5_config = T5Config.from_pretrained(Config.model_name)

t5_model = T5ForConditionalGeneration(t5_config)
t5_model.config.use_cache = False  # no idea why is it True by default, not needed
t5_model.config.max_length = Config.tgt_max_tokens  # needs to be set for inference

tokenizer = T5TokenizerFast.from_pretrained(Config.tokenizer_name)

res = t5_model.load_state_dict(torch.load(Config.checkpoint_name, map_location=torch.device('cpu')))
print(f'Model loading result: {res}')


def inference(model, df_inf, tknzr):
    ds_inf = CNNDailyMailDataset(df_inf, tknzr)
    dl_inf = DataLoader(ds_inf, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    model.to(device)
    model.eval()

    summary_list = []
    for (input_ids, attention_mask, _, _) in tqdm(dl_inf):
        # set data to same device as model
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # run model
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attention_mask,
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95
                                 )
        summaries = tokenizer.batch_decode(out, skip_special_tokens=True)
        summary_list.extend(summaries)

    torch.cuda.empty_cache()
    gc.collect()

    df_generated = df_inf.copy()
    df_generated['generated'] = summary_list
    return df_generated


df_generated = inference(t5_model, df_inf, tokenizer)

print(df_generated.head())
