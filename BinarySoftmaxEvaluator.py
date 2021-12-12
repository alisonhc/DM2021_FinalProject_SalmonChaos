import logging
from typing import List
import numpy as np
import os
import csv

from sentence_transformers import InputExample

logger = logging.getLogger(__name__)

class BinarySoftmaxEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision, precision, recall, and f1
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        self.csv_file = "BinarySoftmaxEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy", "F1", "Precision", "Recall"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("BinarySoftmaxEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        pred_labels = np.argmax(pred_scores, axis=1)
        assert len(pred_labels) == len(self.labels)
        acc, f1, precision, recall = self.get_precision_recall_f1(pred_labels=pred_labels)

        logger.info(f"Accuracy:           {acc * 100}")
        logger.info(f"F1:   {f1 * 100}")
        logger.info(f"Precision:          {precision * 100}")
        logger.info(f"Recall:             {recall * 100}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, f1, precision, recall])

        return f1

    def get_precision_recall_f1(self, pred_labels):
        num_acc = 0
        total_num_false_pos_preds = 0
        total_num_true_pos_preds = 0
        total_num_false_neg_preds = 0

        for i in range(len(pred_labels)):
            if pred_labels[i] == self.labels[i]:
                num_acc += 1
                if pred_labels[i] == 1:
                    total_num_true_pos_preds += 1
            else:
                if pred_labels[i] == 0:
                    total_num_false_neg_preds += 1
                else:
                    total_num_false_pos_preds += 1
        precision = total_num_true_pos_preds / (total_num_true_pos_preds + total_num_false_pos_preds)
        recall = total_num_true_pos_preds / (total_num_true_pos_preds + total_num_false_neg_preds)
        acc = num_acc / len(self.labels)
        f1 = 2 * precision * recall / (precision + recall)
        return acc, f1, precision, recall
