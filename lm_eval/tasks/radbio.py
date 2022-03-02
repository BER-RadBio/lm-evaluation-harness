""""
RadBio AI project - evaluating GPT on our dataset
"""

from lm_eval.base import Task, rf
from pathlib import Path
import pickle
from lm_eval.metrics import mean, perplexity, matthews_corrcoef, f1_score


class RadBio(Task):
    """Base Class for yes/no questions"""
    DATASET_PATH = Path("./radbio_data")
    VERSION = 1.0

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

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


class isInSystemQA(RadBio):

    def download(self):
        pass

    def training_docs(self):
        with open("/homes/mzvyagin/radbio/data/isInSystemQAlarge/train.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open("/homes/mzvyagin/radbio/data/isInSystemQAlarge/test.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError


class goAHumanQA(RadBio):
    def download(self):
        pass

    def training_docs(self):
        return NotImplementedError

    def validation_docs(self):
        return NotImplementedError

    def test_docs(self):
        return NotImplementedError


class goARadiationResponseQA(RadBio):
    def download(self):
        pass

    def training_docs(self):
        return NotImplementedError

    def validation_docs(self):
        return NotImplementedError

    def test_docs(self):
        return NotImplementedError
