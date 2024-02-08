
import torch
from FactCloze.extractor import FactualFactorExtractor
from transformers import AutoTokenizer, AutoModelForMaskedLM


def convert_to_filtered_format_(samples):
    converted_samples = []
    for sample in samples:
        converted_samples.append({
            'document': sample['document'],
            'summary': sample['summary']['sentences'],
            'factors': sample['summary']['factors'],
        })
    return converted_samples


class FactCloze:
    def __init__(self, model_path, tokenizer_path, fact_extractor, mode='BART'):
        self.extractor = FactualFactorExtractor(fact_extractor=fact_extractor, use_gpu=False)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.mode = mode

        print(f'Current Version is "FactCloze-{mode}".')
        print(f'Model path is "{model_path}".')
        print(f'Tokenizer path is "{tokenizer_path}".')

    def correct(self, documents, summaries, do_self_diagnosis=False):

        if self.mode == 'BART':
            from utils_bart import self_diagnosis, filter_factors, correct
        elif self.mode == 'T5':
            from utils_t5 import self_diagnosis, filter_factors, correct
        else:
            raise ValueError('Current version only supports ["BART","T5"]!')

        samples = self.extractor.extract(documents=documents,
                                         summaries=summaries,
                                         k=1,
                                         selection='entity_first',
                                         eval_batch_size=12,
                                         granularity='sentence',
                                         use_tqdm=True)

        if do_self_diagnosis:
            samples = self_diagnosis(self.model, self.tokenizer, samples)
            samples = filter_factors(samples)
        else:
            samples = convert_to_filtered_format_(samples)

        samples = correct(self.model, self.tokenizer, samples)

        return samples


