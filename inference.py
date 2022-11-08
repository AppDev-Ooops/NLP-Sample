import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, TextClassificationPipeline
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained('./results/checkpoint-best').to(device)
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
predictor = TextClassificationPipeline(
                model=model, tokenizer=tokenizer,
                device=-1 if model.device.type == 'cpu' else model.device.index
            )

print(predictor('非常感謝 AppDev Ooops 寫了這麼多好文章！'))
