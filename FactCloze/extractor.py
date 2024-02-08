import importlib
import json
import torch
import spacy

from tqdm import tqdm


def save_to_jsonl(samples, path):
    with open(path, 'w', encoding='utf8') as file:
        for sample in samples:
            file.write(json.dumps(sample) + '\n')


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def convert_word_idx_to_char_idx(seg_summary, factors):
    n_factors = []

    for factor in factors:
        bias = int(factor['start'] > 0)
        start = len(' '.join(seg_summary[:factor['start']])) + bias
        end = len((' '.join(seg_summary[:factor['end']])))

        n_factors.append({
            'text': factor['text'],
            'start': start,
            'end': end
        })

    return n_factors


class FactualFactorExtractor:
    def __init__(self, fact_extractor='en_core_web_trf', use_gpu=False, device=0):

        if use_gpu:
            spacy.require_gpu(device)

        # load nlp tool
        # it can be loaded by "import en_core_web_trf" or "spacy.load('en_core_web_trf')"
        if isinstance(fact_extractor, str):        
            try:
                module = importlib.import_module(fact_extractor)
                self.nlp = module.load()
            except ImportError:
                self.nlp = spacy.load(fact_extractor)
        else:
            self.nlp = fact_extractor


    def extract_factual_factors(self,
                                documents,
                                summaries,
                                eval_batch_size,
                                granularity='sentence',
                                use_tqdm=True):
        """
        1. Split sentences by SpaCy in order to match the index in entities or nouns extracted by it.
        2. Extract factual factors from summary by SpaCy.
        """

        processed_data = []
        processed_documents = self.nlp.pipe(documents, batch_size=eval_batch_size)
        processed_summaries = self.nlp.pipe(summaries, batch_size=eval_batch_size)

        if use_tqdm:
            _bar = tqdm(range(len(documents)), desc=f'Processing factual factors', ncols=150)
            bar = zip(_bar, processed_documents, processed_summaries)
        else:
            # print(f'Processing factual factors from {len(documents)} samples...')
            _bar = range(len(documents))
            bar = zip(_bar, processed_documents, processed_summaries)

        # process each pair of document and summary
        for i, document, summary in bar:

            seg_document = [word.text for word in document]

            bias = 0  # sentence bias
            summary_sentences = []
            summary_entities = []
            summary_nouns = []

            if granularity == 'sentence':
                seg_sentences = summary.sents
            else:
                seg_sentences = [summary]

            for seg_sentence in seg_sentences:
                seg_summary_sentence = [word.text for word in seg_sentence]
                entities = [{'text': ent.text,
                             'start': ent.start - bias,
                             'end': ent.end - bias} for ent in seg_sentence.ents]
                nouns = [{'text': noun_chunk.text,
                          'start': noun_chunk.start - bias,
                          'end': noun_chunk.end - bias} for noun_chunk in
                         seg_sentence.noun_chunks]
                bias += len(seg_summary_sentence)

                # convert the word indices to char indices
                # 'I am a good man.' (good, 3, 4) to (good, 7, 11)
                entities = convert_word_idx_to_char_idx(seg_summary_sentence, entities)
                nouns = convert_word_idx_to_char_idx(seg_summary_sentence, nouns)

                summary_sentences.append(' '.join(seg_summary_sentence))    # merge to a string
                summary_entities.append(entities)
                summary_nouns.append(nouns)

            processed_data.append({
                'document': ' '.join(seg_document),
                'summary_sentences': summary_sentences,
                'summary_entities': summary_entities,
                'summary_nouns': summary_nouns
            })

        return processed_data

    def mixture_factual_factors(self, factors1, factors2):
        """
        Drop out the factors2's factors which overlap factors1's factors.
        """
        candidate_factors = []

        for j in range(len(factors2)):
            for i in range(len(factors1)):
                if i == 0:
                    if factors2[j]['end'] <= factors1[i]['start']:
                        candidate_factors.append(factors2[j])
                        break
                else:
                    if factors2[j]['start'] >= factors1[i - 1]['end'] and factors2[j]['end'] <= factors1[i]['start']:
                        candidate_factors.append(factors2[j])
                        break

            if factors1 and factors2[j]['start'] >= factors1[-1]['end']:
                candidate_factors.extend(factors2[j:])
                break

        if not factors1:
            factors = factors2
        else:
            factors = factors1 + candidate_factors
            factors = list(sorted(factors, key=lambda x: x['start']))

        return factors

    def select_factual_factors(self, entities, nouns, selection):

        if selection == 'entity':
            return entities
        elif selection == 'noun':
            return nouns
        elif selection == 'noun_first':
            return self.mixture_factual_factors(nouns, entities)
        else:
            return self.mixture_factual_factors(entities, nouns)


    def block_factual_factors(self, data, k, selection, use_tqdm):

        processed_data = []

        if use_tqdm:
            bar = tqdm(data, desc=f'Blocking factual factors by k={k}', ncols=150)
        else:
            # print(f'Blocking factual factors by k={k} from {len(data)} samples...')
            bar = data

        for i, sample in enumerate(bar):

            n_sample = {
                'document': sample['document'],
                'summary': {
                    'sentences': [],
                    'factors': []
                }
            }

            for j, (sentence, entities, nouns) in enumerate(
                    zip(sample['summary_sentences'], sample['summary_entities'], sample['summary_nouns'])):
                factors = self.select_factual_factors(entities, nouns, selection)

                n_sample['summary']['sentences'].append(sentence)
                n_sample['summary']['factors'].append(factors)

            processed_data.append(n_sample)

        return processed_data

    def extract(self,
                documents,
                summaries,
                k,
                selection,
                eval_batch_size,
                granularity,
                use_tqdm):
        """
        1. Extract factual factors from summary.
        2. Block the samples by k.
        """
        processed_data = self.extract_factual_factors(documents, summaries, eval_batch_size, granularity, use_tqdm)
        processed_data = self.block_factual_factors(processed_data, k, selection, use_tqdm)

        return processed_data


def get_manually_annotation():

    data = load_jsonl('samples.jsonl')

    samples = []
    for d in data:
        samples.append({
            'document': d['document'],
            'hypothesis': ' '.join(d['hypothesis'])
        })

    return samples


if __name__ == '__main__':

    samples = get_manually_annotation()

    extractor = FactualFactorExtractor(use_gpu=True)
    processed_data = extractor.extract(documents=[sample['document'] for sample in samples],
                                       summaries=[sample['hypothesis'] for sample in samples],
                                       k=1,
                                       selection='entity_first',
                                       eval_batch_size=12,
                                       granularity='sentence',
                                       use_tqdm=True)

    for d in processed_data:
        print(d)

    # save_to_jsonl(processed_data, './processed/human/test_annotation_cnndm.jsonl')


{'document': "It 's well known that exercise can make your muscles bigger . Now , a study has found it may make your brain larger , too . Physical activity can increase grey matter in the brain , increasing the size of areas that contribute to balance and coordination , according to \xa0 Health Day news . The changes in the brain may have health implications in the long - term , such as reducing the risk of falling , said the study 's author , Dr Urho Kujala , of the University of Jyvaskyla . Scroll down for video . Exercise can increase the size of areas of the brain that contribute to balance and coordination , a study found . It could also reduce the risk of being immobile in older age , he added . Dr Kujala said physical activity has already been linked to a number of health benefits , such as lower levels of body fat , reduced heart disease risk factors , better memory and thinking , and a lower risk of type 2 diabetes . But he and his team wanted to understand how exercise affects the brain . They recruited 10 pairs of identical twins , who were all men aged 32 to 36 years . Focusing on twins , who have the same DNA , would allow researchers to see how their environment affects their bodies . In each pair of twins , one brother had exercised more over the past three years than the other , though they reported they carried out similar levels of exercise earlier in their lives . Dr Kujala said : ' On average , the more active members of twin pairs were jogging three hours more per week compared to their inactive co - twins . ' The twins had MRI scans of their brains so researchers could see whether physical activity had any impact on the size of their brains , and specific regions . Exercise did n't seem to affect the size of the brain as a whole , Dr Kujala said . But there was a connection between more activity and more brain volume in areas related to movement , he added . Previous research found exercise is linked to lower levels of body fat , a reduced risk of heart disease , better memory and thinking , and a lower risk of type 2 diabetes . The twins who exercised more did a better job of controlling their blood sugar , which reduces the risk of diabetes , a finding which is already well - known . The study was published in the journal Medicine & Science in Sports & Exercise . It comes after US researchers found regular exercise can also make you smarter . University of South Carolina experts found \xa0 regular treadmill sessions create more mitochondria - \xa0 structures in the cells that produce the body 's energy - in the brain . This energy boost helped the brain to work faster and more efficiently , effectively keeping it younger , researchers said . In the short term this could reduce mental fatigue and sharpen your thinking in between gym sessions . And building up a large reservoir of mitochondria in the brain could also create a ' buffer ' against age - related brain diseases such as Alzheimer 's .",
 'summary': {'sentences':
                 ['Study : Exercising increases the amount of grey matter in the brain .',
                  'It makes areas of the brain that control balance and co - ordination bigger .',
                  'In the long term this could reduce the risk of falling or becoming immobile .',
                  "Previous studies show exercise can stave off Alzheimer 's and diabetes ."],
             'factors': [
                 [{'text': 'Study', 'start': 0, 'end': 5},
                  {'text': 'Exercising', 'start': 8, 'end': 18},
                  {'text': 'the amount', 'start': 29, 'end': 39},
                  {'text': 'grey matter', 'start': 43, 'end': 54},
                  {'text': 'the brain', 'start': 58, 'end': 67}],
                 [{'text': 'It', 'start': 0, 'end': 2},
                  {'text': 'areas', 'start': 9, 'end': 14},
                  {'text': 'the brain', 'start': 18, 'end': 27},
                  {'text': 'that', 'start': 28, 'end': 32},
                  {'text': 'balance', 'start': 41, 'end': 48},
                  {'text': 'co', 'start': 53, 'end': 55},
                  {'text': '-', 'start': 56, 'end': 57},
                  {'text': 'ordination', 'start': 58, 'end': 68}],
                 [{'text': 'the long term', 'start': 3, 'end': 16},
                  {'text': 'this', 'start': 17, 'end': 21},
                  {'text': 'the risk', 'start': 35, 'end': 43}], [{'text': 'Previous studies', 'start': 0, 'end': 16}, {'text': 'exercise', 'start': 22, 'end': 30}, {'text': 'Alzheimer', 'start': 45, 'end': 54}, {'text': 'diabetes', 'start': 62, 'end': 70}]]}}
