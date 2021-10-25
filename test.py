from tqdm import tqdm
import pysolr


class DBSolr:
    """Class for managing Solr database of pipeline outputs"""

    def __init__(self, port, core_name):
        """Set up Solr client"""

        self.solr = pysolr.Solr(f'http://localhost:{port}/solr/{core_name}/', always_commit=True)

    def gather_ocr_texts_and_metadata(self, query):
        """Extract OCR article texts and metadata from Solr database via search"""

        ocr_article_texts = []
        ocr_article_headlines = []
        ocr_article_faids = []
        ocr_article_metadata = []

        self.solr.ping()
        print("Gathering results of Solr search...")
        for doc in tqdm(self.solr.search(query, fl='id,image_file_name,full_article_id,article,headline', sort='id ASC', cursorMark='*')):
            if 'article' in doc:
                ocr_article_texts.extend(doc['article'])
                # ocr_article_headlines.extend(doc['headline'])
                ocr_article_faids.extend(doc['full_article_id'])
                ocr_article_metadata.extend(doc['image_file_name'])

        self.ocr_article_texts = ocr_article_texts
        self.ocr_article_headlines = ocr_article_headlines
        self.ocr_article_faids = ocr_article_faids
        self.ocr_article_metadata = ocr_article_metadata
        self.ocr_article_dates = [extract_date_from_metadata(md) for md in ocr_article_metadata]


def extract_date_from_metadata(metadata):

    fields = metadata.split('-')
    month, day, year = fields[-5:-2]
    return month, int(day), int(year)


if __name__ == '__main__':

    solr_port = 8983
    solr_core_name = 'mytest2'
    years = [1968]

    # create pipeline output object
    db = DBSolr(port=solr_port, core_name=solr_core_name)

    # assemble OCR data from cold pipeline output
    #db.gather_ocr_texts_and_metadata(query='image_file_name:"-1968"')
    db.gather_ocr_texts_and_metadata(query='article:"senate" AND (article:"pill" OR article:"oral" OR '
                                           'article:"contracepti") AND image_file_name:"-1968"')

    print(db.ocr_article_faids)

