
from FactCloze.pipeline import FactCloze
import en_core_web_trf


fact_extractor = en_core_web_trf.load()

corrector = FactCloze(
    model_path="path to the model",
    tokenizer_path="path to the tokenizer",
    fact_extractor=fact_extractor
)

documents = ["Document #1", "Document #2"]
hypotheses = ["Hypothesis #1", "Hypothesis #2"]

reuslts = corrector.correct(documents, hypotheses, do_self_diagnosis=False)
