import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    EvalPrediction
)
from Dataset import ReviewDataset


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

model = AutoModelForSequenceClassification.from_pretrained(
            'hfl/chinese-macbert-base', num_labels=2
        )

train_data = {
    'text': ['超愛這個 App，已經用五六年了！', '難用死了好爛', 
              '希望可以新增桌面小工具的功能', '廣告可以少一點嗎'],
    'label': [0, 0, 1, 1]
}

validation_data = {
    'text': ['每個人都應該要下載 太好用了', '可不可以帳號綁定 FB 就好'],
    'label': [0, 1]
}

train_set = ReviewDataset(train_data)   # 拿來訓練模型參數的資料
dev_set = ReviewDataset(validation_data)     # 訓練過程中衡量模型表現的資料

training_args = TrainingArguments(
    output_dir='./results',      # 把訓練的 model 存在 ./results 目錄中
    learning_rate=5e-5,          # 這就是俗稱的「調參」的一部分 
    num_train_epochs=10,         # 要把訓練資料看過幾輪
    logging_strategy='epoch',    # 
    evaluation_strategy='epoch',
    save_strategy='epoch',
    metric_for_best_model='acc',
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(    
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=dev_set,
    compute_metrics=compute_accuracy
)

trainer.train()
model.save_pretrained('./results/checkpoint-best')