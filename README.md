# bert-seq-cls-avg-pooling
An extended version of BERTForSequenceClassification to use mean pooling.  
It wraps existing `**ForSequenceClassification`.

The following classes are supported now. (Some of the other classes may also work.)
- `BertForSequenceClassification`
- `XLNetForSequenceClassification`
- `RobertaForSequenceClassification`
- `DebertaV2ForSequenceClassification`
- `ElectraForSequenceClassification`

# Install
```sh
pip install git+https://github.com/gotutiyan/bert-seq-cls-avg-pooling
```

# Usage
The interface is the same as `**ForSequenceClassification`.

```python
from seq_cls_mean_pool import AutoModelForSequenceClassificationMeanPool
from transformers import AutoTokenizer
import torch
import torch.optim as optim

models = [
    'google-bert/bert-base-cased',
    'xlnet/xlnet-base-cased',
    'FacebookAI/roberta-base',
    'microsoft/deberta-v3-base',
    'google/electra-large-discriminator'
]
for m in models:
    print(m)
    # You can use .from_pretrained
    model = AutoModelForSequenceClassificationMeanPool.from_pretrained(m, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(m)
    optimizer = optim.Adam(model.parameters())
    sent = ['this is a sample sentence.', 'this code is a sample script.']
    encode = tokenizer(sent, padding=True, return_tensors='pt')
    encode['labels'] = torch.tensor([0, 1])
    output = model(**encode)
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    
    # You can use save_pretrained.
    # The instance of `**ForSequenceClassification` is save (not instance of `AutoModelForSequenceClassificationMeanPool`). 
    model.save_pretrained('sample/')
    # You can load the checkpoint via from_pretrained()
    new_model = AutoModelForSequenceClassificationMeanPool.from_pretrained('sample/')
```