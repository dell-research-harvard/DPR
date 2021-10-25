from tqdm import tqdm
import pysolr
from typing import Dict

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    take_max_roberta_paragraphs
)


class DBSolr:
    """Class for managing Solr database of pipeline outputs"""

    def __init__(self, port, core_name):
        """Set up Solr client"""

        self.solr = pysolr.Solr(f'http://localhost:{port}/solr/{core_name}/', always_commit=True)

    def gather_ocr_texts_and_metadata(self, query):
        """Extract OCR article texts and metadata from Solr database via search"""

        ids = []
        articles = []
        headlines = []

        self.solr.ping()
        print("Gathering results of Solr search...")
        for doc in tqdm(self.solr.search(query, fl='id,article,headline', sort='id ASC', cursorMark='*')):
            ids.append(doc['id'])
            articles.extend(doc['article'])
            headlines.append(doc['headline'][0])

        self.ids = ids
        self.articles = articles
        self.headlines = headlines


class NewspaperArchiveCtxSrc_heads_solr:  # Needs to inherit from RetrieverData
    def __init__(
            self,
            solr_port: int,
            solr_core_name: str,
            years: list = [],
            normalize: bool = False,
    ):
        self.solr_port = solr_port
        self.solr_core_name = solr_core_name
        self.years = years
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):

        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        print("Creating bi-encoder dict...")

        # create pipeline output object
        db = DBSolr(port=solr_port, core_name=solr_core_name)

        # assemble OCR data from cold pipeline output
        # db.gather_ocr_texts_and_metadata(query='image_file_name:"-1968"')
        db.gather_ocr_texts_and_metadata(
            query='headline:"senate" AND (article:"pill" OR article:"oral" OR '                                         # Random small search
                  'article:"contracepti") AND image_file_name:"-1968"')

        for i in range(len(db.ids)):
            uid = db.ids[i]
            if self.normalize:
                title = normalize_passage(db.headlines[i])
                title = title.lower()
                passage = take_max_roberta_paragraphs(db.articles[i], title, tokenizer)
                passage = normalize_passage(passage)
            else:
                title = db.headlines[i]
                passage = db.articles[i]
            ctxs[uid] = BiEncoderPassage(passage, title)


if __name__ == '__main__':
    solr_port = 8983
    solr_core_name = 'mytest2'
    years = [1968]

    # create pipeline output object
    ctx_class = NewspaperArchiveCtxSrc_heads_solr(solr_port, solr_core_name, years, True)

    ctxs = {}

    ctx_class.load_data_to(ctxs)

    print(ctxs)
