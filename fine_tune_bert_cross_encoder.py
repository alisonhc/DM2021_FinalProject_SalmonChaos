import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CESoftmaxAccuracyEvaluator
from BinarySoftmaxEvaluator import BinarySoftmaxEvaluator
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
#from google.colab import files

MODEL_NAME = 'hfl/chinese-macbert-base'

batch_size = 4

model = CrossEncoder(MODEL_NAME, num_labels=2, max_length=512, device='cuda:1')
#class_weights = torch.tensor()

# from google.colab import drive
# drive.mount('/content/drive')

def get_examples(fpath):
  examples = []
  fname = fpath.split('/')[-1].split('.')[-2]
  gu_id = 0
  for line in open(fpath, encoding="utf-8"):
    splits = line.strip().split('\t')
    label = int(splits[2])
    assert label == 0 or label == 1
    text1 = splits[0]
    text2 = splits[1]
    guid = "%s-%d" % (fname, gu_id)
    gu_id += 1
    examples.append(InputExample(guid=guid, texts=[text1, text2], label=label))
  return examples

train_pair_examples = get_examples('train_pairs_imbalanced10_rep.tsv')

print(len(train_pair_examples))
print(train_pair_examples[:3])

def get_pairs_label_lists(fpath):
  pairs = []
  labels = []
  label_0 = 0
  label_1 = 0
  for line in open(fpath, encoding="utf-8"):
    splits = line.strip().split('\t')
    label = int(splits[2])
    assert label == 0 or label == 1
    if label == 0:
      label_0 += 1
    elif label == 1:
      label_1 += 1
    text1 = splits[0]
    text2 = splits[1]
    pairs.append([text1, text2])
    labels.append(label)
  print(label_0, label_1)
  return pairs, labels

def calculate_class_weights(fpath):
  label_0 = 0
  label_1 = 0
  for line in open(fpath, encoding="utf-8"):
    splits = line.strip().split('\t')
    label = int(splits[2])
    if label == 0:
      label_0 += 1
    if label == 1:
      label_1 += 1
  print(label_0, label_1)
  normed_weights = [1/label_0, 1/label_1]
  normed_weights = torch.FloatTensor(normed_weights).to('cuda:1')
  return normed_weights

train_weights = calculate_class_weights('train_pairs_imbalanced10_rep.tsv')
#print(train_weights)
train_data_loader = DataLoader(train_pair_examples, batch_size=batch_size, shuffle=True)

pairs, labels = get_pairs_label_lists('test_pairs_imbalanced10_rep.tsv')
print(len(pairs))

#ce_loss = CrossEntropyLoss(weight=train_weights)

evaluator = BinarySoftmaxEvaluator(pairs, labels)
#evaluator = CESoftmaxAccuracyEvaluator(pairs, labels)

model_save_path = 'macbert_ce_sigmoid_rep_split10_imbalanced_grad_accum'

model.fit(train_dataloader=train_data_loader, epochs=2,
          output_path=model_save_path, evaluator=evaluator) #, loss_fct=ce_loss)
