from tqdm import tqdm
import pysolr


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
            ids.extend(doc['id'])
            articles.extend(doc['article'])
            #headlines.extend(doc['headline'])

        self.ids = ids
        self.articles = articles
        self.headlines = headlines


if __name__ == '__main__':

    solr_port = 8983
    solr_core_name = 'mytest2'
    years = [1968]

    # create pipeline output object
    db = DBSolr(port=solr_port, core_name=solr_core_name)

    # assemble OCR data from cold pipeline output
    #db.gather_ocr_texts_and_metadata(query='image_file_name:"-1968"')
    db.gather_ocr_texts_and_metadata(query='headline:"senate" AND (article:"pill" OR article:"oral" OR '                 # Random small search
                                           'article:"contracepti") AND image_file_name:"-1968"')

    print(len(db.articles))

    print(db.articles)
