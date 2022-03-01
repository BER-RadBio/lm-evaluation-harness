""""
RadBio AI project - evaluating GPT on our dataset
"""

from lm_eval.base import Task, rf
from pathlib import Path
import pickle
from lm_eval.metrics import mean, perplexity

class RadBio(Task):
    # insert path name here
    # DATASET_PATH = Path("/homes/mzvyagin/isInSystemQA.obj")

    def download(self):
        # some kind of unpickling call here? I don't think we can download from the internet as it's not public
        pass

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        # These should return a Python iterable (list or generator) of dicts that can be queried for individual doc
        # examples. NOTE: If your task doesn't have a train/validation/test set, remember to raise a NotImplementedError
        # for that specific split.
        # load in the training data from the pickle and return as iterable
        with open("/homes/mzvyagin/isInSystemQA.obj", "rb") as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        return NotImplementedError

    def test_docs(self):
        return NotImplementedError

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))
        return ll, is_greedy

    def process_results(self, doc, results):
        ll, is_greedy = results

        return {
            'ppl': ll,
            'acc': int(is_greedy)
        }

    def aggregation(self):
        return {
            'ppl': perplexity,
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'ppl': False,
            'acc': True
        }