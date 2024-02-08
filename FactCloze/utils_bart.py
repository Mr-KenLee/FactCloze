import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def clean(text):
    end = None if '<pad>' not in text else text.index('<pad>')
    text = text[:end]
    text = text.replace('<s>', '')
    text = text.replace('</s>', '')
    return text


def convert_input_format(batch, tokenizer):
    documents = [b['document'] for b in batch]
    summaries = [b['masked_summary'] for b in batch]
    targets = [b['target'] for b in batch]

    encoder_outputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(documents, summaries)),
        padding=True,
        max_length=1024,
        truncation=True,
        return_tensors='pt')

    decoder_outputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=targets,
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors='pt')

    input_ids, attention_mask = encoder_outputs['input_ids'], encoder_outputs['attention_mask']
    labels, decoder_attention_mask = decoder_outputs['input_ids'], decoder_outputs['attention_mask']

    return input_ids.cuda(), attention_mask.cuda(), decoder_attention_mask.cuda(), labels.cuda()


def self_diagnosis(model, tokenizer, samples):

    def _convert(_samples):
        final = []
        for i, d in enumerate(_samples):
            for j, (sentence, factors) in enumerate(zip(d['summary']['sentences'], d['summary']['factors'])):
                for factor in factors:

                    start, end = factor['start'], factor['end']
                    masked_summary = sentence[:start] + tokenizer.mask_token + sentence[end:]

                    final.append({
                        'doc_id': i,
                        'sent_id': j,
                        'document': d['document'],
                        'masked_summary': masked_summary,
                        'target': sentence,
                        'factor': factor
                    })
        return final

    samples = _convert(samples)
    loader = DataLoader(dataset=MyDataset(samples), shuffle=False, batch_size=4, collate_fn=lambda x: x)

    predictions = []

    model.eval()

    with torch.no_grad():

        bar = tqdm(loader, ncols=150, desc='self diagnosis')
        for i, batch in enumerate(bar):
            input_ids, attention_mask, decoder_attention_mask, labels = convert_input_format(batch, tokenizer)

            # transformers库
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                do_sample=False,
            )

            candidates = tokenizer.batch_decode(
                sequences=generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            candidates = [clean(candidate) for candidate in candidates]

            for candidate, sample in zip(candidates, batch):
                predictions.append({
                    **sample,
                    'corrected': candidate
                })

    results = {}
    for prediction in predictions:

        doc_id = prediction['doc_id']
        sent_id = prediction['sent_id']

        if doc_id not in results:
            results[doc_id] = {
                'document': prediction['document'],
                'summary': {}
            }
        if sent_id not in results[doc_id]['summary']:
            results[doc_id]['summary'][sent_id] = []

        results[doc_id]['summary'][sent_id].append({
            'sentence': prediction['target'],
            'corrected': prediction['corrected'],
            'factor': prediction['factor']
        })

    results = [value for value in results.values()]

    return results


def filter_factors(samples):

    filtered = []
    for sample in samples:
        document = sample['document']
        summary = []
        factors = []

        for info in sample['summary'].values():
            sentence = ''
            _factors = []
            for single in info:
                sentence = single['sentence']
                corrected = single['corrected']

                if single['factor'] is not None and sentence.lower() != corrected.lower():
                    _factors.append(single['factor'])

            summary.append(sentence)
            factors.append(_factors)

        filtered.append({
            'document': document,
            'summary': summary,
            'factors': factors
        })

    return filtered


def correct(model, tokenizer, samples):

    def _convert(_samples):
        final = []

        for i, d in enumerate(_samples):
            for j, (sentence, factors) in enumerate(zip(d['summary'], d['factors'])):

                _factors = sorted(factors, key=lambda x:x['start'])
                masked_summary = ''
                p = 0

                for factor in factors:
                    start, end = factor['start'], factor['end']
                    masked_summary += sentence[p:start] + tokenizer.mask_token
                    p = end

                masked_summary += sentence[p:]

                final.append({
                    'doc_id': i,
                    'document': d['document'],
                    'masked_summary': sentence if not masked_summary else masked_summary,
                    'target': sentence,
                    'factors': _factors
                })

        return final

    samples = _convert(samples)
    print(samples[0]['masked_summary'])
    loader = DataLoader(dataset=MyDataset(samples), shuffle=False, batch_size=4, collate_fn=lambda x: x)

    predictions = []

    model.eval()

    with torch.no_grad():

        bar = tqdm(loader, ncols=150, desc='correct')
        for i, batch in enumerate(bar):
            input_ids, attention_mask, decoder_attention_mask, labels = convert_input_format(batch, tokenizer)

            # transformers库
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                do_sample=False,
            )

            candidates = tokenizer.batch_decode(
                sequences=generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            candidates = [clean(candidate) for candidate in candidates]

            for candidate, sample in zip(candidates, batch):
                predictions.append({
                    **sample,
                    'corrected': candidate
                })

    results = {}
    for prediction in predictions:

        doc_id = prediction['doc_id']

        if doc_id not in results:
            results[doc_id] = {
                'document': prediction['document'],
                'summary': [],
                'corrected': [],
                'factors': [],
            }

        results[doc_id]['summary'].append(prediction['target'])
        results[doc_id]['corrected'].append(prediction['corrected'])
        results[doc_id]['factors'].append(prediction['factors'])

    results = [value for value in results.values()]

    return results












