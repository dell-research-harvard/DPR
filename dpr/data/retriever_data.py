import collections
import csv
import json
import logging
import pickle
from typing import Dict
import random

import hydra
import jsonlines
import torch
from omegaconf import DictConfig
import ijson
import glob
from tqdm import tqdm
from transformers import BartTokenizerFast
from datetime import datetime

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    normalize_question,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)

logger = logging.getLogger(__name__)
QASample = collections.namedtuple("QuerySample", ["query", "id", "answers"])
TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(
            self.data_files
        )
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(
            self.query_special_suffix
        ):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col

    def load_data(self):
        super().load_data()
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class CustomQASrc(QASrc):
    def __init__(
        self,
        question: int,
        answer: int,
        id: int,
    ):
        self.data = None
        self.selector = None
        self.special_query_token = None
        self.query_special_suffix = None
        self.question = question
        self.answer = tuple([answer])
        self.id = id

    def load_data(self):
        data = []
        data.append(QASample(self._process_question(self.question), self.id, self.answer))
        self.data = data


class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class CustomCsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file_path: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize
        self.file_path = file_path

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        with open(self.file_path) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class NewspaperArchiveCtxSrc(RetrieverData):
    def __init__(
        self,
        path_pattern: str,
        layout_object: str = 'article',
        page_filter: int = None,
        id_prefix: str = None,
        normalize: bool = False,
        n_random_papers: bool = False,
    ):
        self.id_prefix = id_prefix
        self.normalize = normalize
        self.file_paths = glob.glob(path_pattern)
        self.layout_object = layout_object
        self.page_filter = page_filter
        self.n_random_papers = n_random_papers

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):

        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        if self.n_random_papers:
            print("Random newspaper subset...")
            scan_names = []
            for file_path in tqdm(self.file_paths):
                with open(file_path, 'rb') as f:
                    items = ijson.kvitems(f, '')
                    for k, v in items:
                        scan_names.append(k)
            papers = list(set([self.get_paper_name(scan) for scan in scan_names]))
            print(f"{len(papers)} total papers...")
            random_papers = random.sample(papers, self.n_random_papers)
            print(f"Selected random papers: {random_papers}")

        print("Creating bi-encoder dict...")
        for file_path in tqdm(self.file_paths):
            
            with open(file_path, 'rb') as f:
                items = ijson.kvitems(f, '')
                ocr_text_generators = []
                for k, v in items:
                    if self.n_random_papers:
                        if self.get_paper_name(k) in random_papers:
                            ocr_text_generators.append(self.ocr_text_iter(v))
                    else:
                        ocr_text_generators.append(self.ocr_text_iter(v))

            if len(ocr_text_generators) == 0:
                continue
                
            for gen in ocr_text_generators:
                for layobj in gen:
                    title, passage, object_id = layobj
                    uid = str(object_id) + '_' + title 
                    if self.normalize:
                        if self.layout_object == 'headline':
                            passage = normalize_passage(passage)
                            passage = passage.lower()
                        else:
                            passage = take_max_model_paragraphs(passage, tokenizer)
                            passage = normalize_passage(passage)
                    ctxs[uid] = BiEncoderPassage(passage, title)

    def ocr_text_iter(self, v):
        for ik in v:
            if ik['label'] == self.layout_object:
                if self.page_filter:
                    if not ik['image_file_name'].split('.')[0].endswith(f'p-{self.page_filter}'):
                        yield (ik['image_file_name'], ik['ocr_text'], ik['object_id'])                                  # defines title, passage, object_id
                else:
                    yield (ik['image_file_name'], ik['ocr_text'], ik['object_id'])

    @staticmethod
    def get_paper_name(file_end):
        return "-".join(file_end.split("-")[1:-5])


class NewspaperArchiveCtxSrc_heads(RetrieverData):

    def __init__(
            self,
            path_pattern: str,
            normalize: bool = False,
            id_prefix: str = None,
            n_random_papers: bool = False,
            month: str = None,
    ):
        self.normalize = normalize
        self.file_paths = glob.glob(path_pattern)
        self.id_prefix = id_prefix
        self.n_random_papers = n_random_papers
        self.month_str = "-" + month + "-"

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):

        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        if self.n_random_papers:
            print("Random newspaper subset...")
            scan_names = []
            for file_path in tqdm(self.file_paths):
                with open(file_path, 'rb') as f:
                    items = ijson.kvitems(f, '')
                    for k, v in items:
                        scan_names.append(k)
            papers = list(set([self.get_paper_name(scan) for scan in scan_names]))
            papers.sort()
            print(f"{len(papers)} total papers...")

            random.seed(789)
            random_papers = random.sample(papers, self.n_random_papers)
            print(f"Selected random papers: {random_papers}")

        print("Creating bi-encoder dict...")
        for file_path in tqdm(self.file_paths):

            with open(file_path, 'rb') as f:
                items = ijson.kvitems(f, '')
                ocr_text_generators = []
                for k, v in items:
                    if self.month_str:
                        if self.month_str in k:
                            if self.n_random_papers:
                                if self.get_paper_name(k) in random_papers:
                                    ocr_text_generators.append(self.ocr_text_iter(v))
                            else:
                                ocr_text_generators.append(self.ocr_text_iter(v))
                    else:
                        if self.n_random_papers:
                            if self.get_paper_name(k) in random_papers:
                                ocr_text_generators.append(self.ocr_text_iter(v))
                        else:
                            ocr_text_generators.append(self.ocr_text_iter(v))

            if len(ocr_text_generators) == 0:
                continue

            for gen in ocr_text_generators:
                for layobj in gen:
                    title, passage, object_id = layobj
                    uid = object_id
                    if self.normalize:
                        title = normalize_passage(title)
                        title = title.lower()
                        passage = take_max_model_paragraphs(passage, tokenizer)
                        passage = normalize_passage(passage)
                    ctxs[uid] = BiEncoderPassage(passage, title)

    def ocr_text_iter(self, v):
        for ik in v:
            yield (ik['headline'], ik['article'], ik['id'])

    @staticmethod
    def get_paper_name(file_end):
        return "-".join(file_end.split("-")[1:-5])


class NewspaperArchiveCtxSrc_heads_daily(RetrieverData):

    def __init__(
            self,
            path_pattern: str,
            id_prefix: str = None,
    ):
        self.file_paths = glob.glob(path_pattern)
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage], date):

        year = str(datetime.strptime(date, "%b-%d-%Y").year)

        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        print(f"Creating bi-encoder dict for {date}...")
        for file_path in tqdm(self.file_paths):

            if year in file_path:
                with open(file_path, 'rb') as f:
                    items = ijson.kvitems(f, '')
                    ocr_text_generators = []
                    for k, v in items:
                        if date in k:
                            ocr_text_generators.append(self.ocr_text_iter(v))

                if len(ocr_text_generators) == 0:
                    continue

                for gen in ocr_text_generators:
                    for layobj in gen:
                        title, passage, object_id = layobj
                        uid = object_id
                        title = normalize_passage(title)
                        title = title.lower()
                        passage = take_max_model_paragraphs(passage, tokenizer)
                        passage = normalize_passage(passage)
                        ctxs[uid] = BiEncoderPassage(passage, title)

    def load_data_from(self, ctxs: Dict[object, BiEncoderPassage], path):

        # for path in list_of_paths:
        date = path.split("/")[-1].split("_")[1]
        year = str(datetime.strptime(date, "%b-%d-%Y").year)

        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        print(f"Creating bi-encoder dict for {date}...")
        for file_path in tqdm(self.file_paths):

            if year in file_path:
                with open(file_path, 'rb') as f:
                    items = ijson.kvitems(f, '')
                    ocr_text_generators = []
                    for k, v in items:
                        if date in k:
                            ocr_text_generators.append(self.ocr_text_iter(v))

                if len(ocr_text_generators) == 0:
                    continue

                for gen in ocr_text_generators:
                    for layobj in gen:
                        title, passage, object_id = layobj
                        uid = object_id
                        title = normalize_passage(title)
                        title = title.lower()
                        passage = take_max_model_paragraphs(passage, tokenizer)
                        passage = normalize_passage(passage)
                        ctxs[uid] = BiEncoderPassage(passage, title)


    @staticmethod
    def ocr_text_iter(v):
        for ik in v:
            yield (ik['headline'], ik['article'], ik['id'])

    @staticmethod
    def get_paper_name(file_end):
        return "-".join(file_end.split("-")[1:-5])


class MnliJsonlCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        passage_char_max: int,
        hypotheses: bool = True,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        self.id_prefix = id_prefix
        self.normalize = normalize
        self.file = file
        self.passage_char_max = passage_char_max
        self.hypotheses = hypotheses

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        with jsonlines.open(self.file, mode="r") as jsonl_reader:

            if self.hypotheses:
                for jline in jsonl_reader:
                    for k in ['positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']:
                        uid = jline[k][0]['title']
                        passage = jline[k][0]['text']
                        if self.normalize:
                            passage = normalize_passage(passage)  
                        ctxs[uid] = BiEncoderPassage(passage[:self.passage_char_max], uid)
            else:
                for jline in jsonl_reader:
                    uid = jline['positive_ctxs'][0]['title'][:-1]
                    passage = jline['question']
                    if self.normalize:
                        passage = normalize_passage(passage)  
                    ctxs[uid] = BiEncoderPassage(passage[:self.passage_char_max], uid)


class HFDatasetsCtxSrc(RetrieverData):
    def __init__(
        self,
        dataset: str,
        split: str,
        passage_char_max: int,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        self.id_prefix = id_prefix
        self.normalize = normalize
        self.dataset = dataset
        self.split = split
        self.passage_char_max = passage_char_max

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):

        from datasets import load_dataset
        hfdataset = load_dataset(self.dataset, split=self.split)

        for idx, spl in enumerate(hfdataset):
            uid = str(idx) + '-class' + str(spl['label'])
            passage = spl['text']
            if self.normalize:
                passage = normalize_passage(passage)  
            ctxs[uid] = BiEncoderPassage(passage[:self.passage_char_max], uid)


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            file, id_col, text_col, title_col, id_prefix, normalize=normalize
        )
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(
            tables_dict, self.tables_chunk_sz, split_type=self.split_type
        )
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)


def take_max_model_paragraphs(ctx_text, tokenizer, tok_space=510, tok_max=512):

    paragraphs = ctx_text.split('\n\n')
    returned_paragraphs = []
    for paragraph in paragraphs:
        para_tokens = tokenizer(paragraph)['input_ids']
        n_tok = len(para_tokens) - 2 + 1
        tok_space -= n_tok
        if tok_space <= 0 and len(returned_paragraphs) == 0:
            return tokenizer.decode(para_tokens[1:tok_max])
        elif tok_space <= 0:
            return "\n".join(returned_paragraphs)
        else:
            returned_paragraphs.append(paragraph)
    return "\n".join(returned_paragraphs)

