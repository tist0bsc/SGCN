# coding: utf-8

import json
from train import train 

with open(r'train_config.json', encoding="utf-8") as f:
    config = json.load(f)

train(config)

