# Python 3.7.11

"""
 Pipeline to match full text back to retriever outputs.
"""

import glob
from tqdm import tqdm
import json
import ijson
import simplejson
import os
import random
import re

n_per_strata = 20
random_sample = False
top_0_1 = False


# Vietnam on Emily
original_data = 'C:/Users/Emily/Documents/Predoc/Newspapers/100_full_paper_samp_full_210929/*'
retrieved_data = 'C:/Users/Emily/Documents/Predoc/Newspapers/newspapers211001b/strata_0_0.json'
save_dir = 'C:/Users/Emily/Documents/Predoc/Newspapers/matched_retriever_output/'

# BarbS on Emily
# original_data = 'C:/Users/Emily/Documents/Predoc/Newspapers/BarbS/inputs/*'
# retrieved_data = 'C:/Users/Emily/Documents/Predoc/Newspapers/BarbS/q4/*'
# save_dir = 'C:/Users/Emily/Documents/Predoc/Newspapers/Barbs/Labelling/'

# Vietnam on Guppy
# original_data = '/mnt/data02/retrieval/retrieval_test/100_full_paper_samp_full_210929/'
# retrieved_data = '/mnt/data02/retrieval/retrieval_test/test_results/newspapers211001b'
# save_dir = '/mnt/data02/retrieval/retrieval_test/matched_retriever_output/'

# Index original data
ids = []
image_file_names = []
image_paths = []
object_ids = []
full_headlines = []
full_articles = []
bboxs = []
full_article_ids = []

for path in tqdm(glob.glob(original_data)):
    with open(path, 'rb') as f:
        items = ijson.kvitems(f, '')
        for k, v in items:
            for i in range(len(v)):
                ids.append(v[i]['id'])
                image_file_names.append(v[i]['image_file_name'])
                image_paths.append(v[i]['image_path'])
                object_ids.append(v[i]['object_id'])
                full_headlines.append(v[i]['headline'])
                full_articles.append(v[i]['article'])
                bboxs.append(v[i]['bbox'])
                full_article_ids.append(v[i]['full_article_id'])

# match to retrieved data
# if random_sample:
sample = []

for path in tqdm(glob.glob(retrieved_data)):
    with open(path, 'rb') as f:
        strata = json.load(f)

        new_strata = []
        strata_num = path.split('\\')[-1:]

        articles = strata[0]['ctxs']

        if top_0_1:
            articles = strata[0]['ctxs'][:860]

        if random_sample:
            articles = random.sample(articles, n_per_strata)

        for art in articles:
            #if not re.search("Jan-\d\d-1970", art['id']):
            #    continue

            index = ids.index(art['id'])

            # Match and format for label studio
            ls_art = {"id": art["id"],
                      "data": {
                          "embedded_headline": art["embedded_title"],
                          "embedded_article": art["embedded_text"],
                          "score": art["score"],
                          "image_file_name": image_file_names[index],
                          "image_path": image_paths[index],
                          "object_id": object_ids[index],
                          "full_headline": full_headlines[index],
                          "full_article": full_articles[index],
                          "bbox": bboxs[index],
                          "full_article_id": full_article_ids[index],
                          "strata": strata_num
                      }
                      }
            # if random_sample:
            #     sample.append(ls_art)
            # else:
            #     new_strata.append(ls_art)
            sample.append(ls_art)

        # if not random_sample:
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_end = path.split('\\')[-1:]                                                                           # might need changing for Guppy
        #     save_path = os.path.join(save_dir, save_end[0])
        #
        #     with open(save_path, "w") as writer:
        #         writer.write(simplejson.dumps(new_strata, indent=4) + "\n")

# if random_sample:
    with open(f'{save_dir}/sample_for_ZSC_sentiment.json', 'w') as writer:
        writer.write(simplejson.dumps(sample, indent=4) + "\n")
