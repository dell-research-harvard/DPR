from tqdm import tqdm
import pysolr
from typing import Dict
import random

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
            n_random_papers: bool = False
    ):
        self.solr_port = solr_port
        self.solr_core_name = solr_core_name
        self.years = years
        self.normalize = normalize
        self.n_random_papers = n_random_papers

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):

        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        # Create solr output object
        db = DBSolr(port=solr_port, core_name=solr_core_name)

        # query_list = []
        # for year in years:
        #     query = f'image_file_name:"-{year}"'
        #     query_list.append(query)
        # search_term = " OR ".join(query_list)

        search_term = 'headline:"senate" AND (article:"pill" OR article:"oral" OR article:"contracepti") AND image_file_name:"-1968"'

        # Gather data from solr
        db.gather_ocr_texts_and_metadata(query=search_term)

        if self.n_random_papers:
            print("Random newspaper subset...")

            papers = list(set([self.get_paper_name(scan) for scan in db.ids]))
            print(f"{len(papers)} total papers...")

            random.seed(789)
            random_papers = random.sample(papers, self.n_random_papers)
            print(f"Selected random papers: {random_papers}")

        print("Creating bi-encoder dict...")
        for i in range(len(db.ids)):

            if self.n_random_papers:
                if self.get_paper_name(db.ids[i]) in random_papers:
                    uid = db.ids[i]
                    title = db.headlines[i]
                    passage = db.articles[i]
                else:
                    continue
            else:
                uid = db.ids[i]
                title = db.headlines[i]
                passage = db.articles[i]

            if self.normalize:
                title = normalize_passage(title)
                title = title.lower()
                passage = take_max_roberta_paragraphs(passage, title, tokenizer)
                passage = normalize_passage(passage)

            ctxs[uid] = BiEncoderPassage(passage, title)

    @staticmethod
    def get_paper_name(image_file_name):
        return "-".join(image_file_name.split("-")[1:-5])


if __name__ == '__main__':
    solr_port = 8983
    solr_core_name = 'mytest2'
    years = [1968]
    n_random_papers = 2

    # create pipeline output object
    ctx_class = NewspaperArchiveCtxSrc_heads_solr(solr_port, solr_core_name, years, True, False)

    ctxs = {}

    ctx_class.load_data_to(ctxs)

    print(ctxs)
