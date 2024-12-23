# bert-seq-cls-avg-pooling
An extended version of BERTForSequenceClassification to use mean pooling.  
It wraps existing `**ForSequenceClassification`.

The following classes are supported now. (Some of the other classes may also work.)
- `BertForSequenceClassification`
- `XLNetForSequenceClassification`
- `RobertaForSequenceClassification`
- `DebertaV2ForSequenceClassification`
- `ElectraForSequenceClassification`
- `XLMRobertaForSequenceClassification`

# Install
```sh
pip install git+https://github.com/gotutiyan/bert-seq-cls-avg-pooling
```

# Usage

### Loading
- Case 1: Initizalize by giving an instance of `AutoModelForSequenceClassification`.

```python
from seq_cls_mean_pool import AutoModelForSequenceClassificationMeanPool
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased')
model = AutoModelForSequenceClassificationMeanPool(base_model)
```

- Case 2: Use `from_pretrained()`.

```python
from seq_cls_mean_pool import AutoModelForSequenceClassificationMeanPool

model = AutoModelForSequenceClassificationMeanPool.from_pretrained('google-bert/bert-base-uncased')
```

### Forwarding
The interface is the same as `**ForSequenceClassification`.

```python
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from seq_cls_mean_pool import AutoModelForSequenceClassificationMeanPool
import torch
import torch.optim as optim

model = AutoModelForSequenceClassificationMeanPool.from_pretrained(
    'google-bert/bert-base-cased',
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
optimizer = optim.Adam(model.parameters())
sent = ['this is a sample sentence.', 'this is a sample script.']
encode = tokenizer(sent, padding=True, return_tensors='pt')
encode['labels'] = torch.tensor([0, 1])
output: SequenceClassifierOutput = model(**encode)

# For training
optimizer.zero_grad()
output.loss.backward()
optimizer.step()

# For predicting
predictions = torch.argmax(output.logits, dim=-1)
    
# You can use save_pretrained.
# The instance of `**ForSequenceClassification` is save (not instance of `AutoModelForSequenceClassificationMeanPool`). 
model.save_pretrained('sample/')
```