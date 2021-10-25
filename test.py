from tqdm import tqdm
import pysolr
from typing import Dict

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    take_max_roberta_paragraphs
)

from dpr.data.retriever_data import (
    RetrieverData
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
            if 'headline' in doc:
                headlines.append(doc['headline'][0])
            else:
                headlines.append("")

        assert len(ids) == len(articles) == len(headlines)

        self.ids = ids
        self.articles = articles
        self.headlines = headlines


class NewspaperArchiveCtxSrc_heads_solr(RetrieverData):
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

        # Create solr output object
        db = DBSolr(port=solr_port, core_name=solr_core_name)

        # Gather data from solr
        query_list = []
        for year in years:
            query = f'image_file_name:"-{year}"'
            query_list.append(query)
        search_term = " OR ".join(query_list)

        db.gather_ocr_texts_and_metadata(query=search_term)

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
