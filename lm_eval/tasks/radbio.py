""""
RadBio AI project - evaluating GPT on our dataset
"""

from lm_eval.base import Task, rf
from pathlib import Path
import pickle
from lm_eval.metrics import mean, perplexity, matthews_corrcoef, f1_score

class RadBio(Task):
    # insert path name here
    # DATASET_PATH = Path("/homes/mzvyagin/isInSystemQA.obj")
    VERSION=1.0

    def download(self):
        # some kind of unpickling call here? I don't think we can download from the internet as it's not public
        pass

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
    #     # These should return a Python iterable (list or generator) of dicts that can be queried for individual doc
    #     # examples. NOTE: If your task doesn't have a train/validation/test set, remember to raise a NotImplementedError
    #     # for that specific split.
    #     # load in the training data from the pickle and return as iterable
    #     with open("/homes/mzvyagin/radbio/isInSystemQA.obj", "rb") as f:
    #         data = pickle.load(f)
    #     split_point = int(len(data) * 0.8)
    #     return data[:split_point]
        with open("/homes/mzvyagin/radbio/isInSystemQA.obj", "rb") as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open("/homes/mzvyagin/radbio/isInSystemQA.obj", "rb") as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        if doc["answer"].strip() == "yes":
            gold = 1
        else:
            gold = 0
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "f1": f1_score
        }