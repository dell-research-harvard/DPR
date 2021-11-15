# Python 3.7.11

import glob
from tqdm import tqdm
import json
import ijson
import simplejson
import os


# Vietnam on Emily
# original_data = 'C:/Users/Emily/Documents/Predoc/Newspapers/all_scans_dbx_text_files_1968_full_bl_211112'
# strata_0_file = 'C:/Users/Emily/Documents/Predoc/Newspapers/newspapers211001b/strata_0_0.json'
# image_size_file = 'C:/Users/Emily/Documents/Predoc/Newspapers/size_dict.json'
# save_dir = 'C:/Users/Emily/Documents/Predoc/Newspapers/sc_matched_retriever_output/'

# # Vietnam on Guppy
original_data = '/mnt/data02/retrieval/preprocess/all_scans_dbx_text_files_1968_full_bl_211112/'
strata_0_file = '/mnt/data02/retrieval/retrieval_test/test_results/newspapers211001b/strata_0_0.json'
image_size_file = '/mnt/data02/retrieval/retrieval_test/size_dict.json'
save_dir = '/mnt/data02/retrieval/retrieval_test/sc_matched_retriever_output/'


query_list = ["accnd_140+c-l-sulzberger-",
              "accnd_140+frank-mankiewicz-",
              "syndicated_columnists+carl-rowan",
              "syndicated_columnists+carl-t-rowan",
              "syndicated_columnists+david-lawrence",
              "syndicated_columnists+holmes-alexander",
              "syndicated_columnists+jack-anderson",
              "syndicated_columnists+james-j-kilpatrick",
              "syndicated_columnists+james-kilpatrick",
              "syndicated_columnists+james-reston",
              "syndicated_columnists+joseph-alsop",
              "syndicated_columnists+joseph-kraft",
              "syndicated_columnists+marquis-childs",
              "syndicated_columnists+ralph-de-toledano",
              "syndicated_columnists+robert-d-novak",
              "syndicated_columnists+robert-novak",
              "syndicated_columnists+roscoe-drummond",
              "syndicated_columnists+rowland-evans",
              "syndicated_columnists+thomas-braden",
              "syndicated_columnists+tom-braden",
              "syndicated_columnists+tom-wicker",
              "syndicated_columnists+william-buckley",
              "syndicated_columnists+william-f-buckley"]


# Index strata 0
scans_in_s0 = []

with open(strata_0_file, 'rb') as f:
    strata_0 = json.load(f)
    articles = strata_0[0]['ctxs']

    for art in articles:
        scan_name = art['id'].split("_")[1]
        scans_in_s0.append(scan_name)

# Get images with sizes
with open(image_size_file, 'rb') as f:
    sizes = json.load(f)
    images_with_sizes = [s + ".jpg" for s in list(sizes.keys())]

# Get other data
counter = 0
for path in tqdm(glob.glob(f'{original_data}/**/*.json')):
    with open(path, 'rb') as f:
        items = ijson.kvitems(f, '')
        for k, v in items:

            if k in scans_in_s0 and k in images_with_sizes:
                for article in v:
                    if any(query in article["query"] for query in query_list):

                        # Create box for full article
                        scan_width = sizes[k.replace(".jpg", "")][0]
                        scan_height = sizes[k.replace(".jpg", "")][1]
                        x = (article['bbox'][0] * 100) / scan_width
                        y = (article['bbox'][1] * 100) / scan_height
                        w = ((article['bbox'][2] - article['bbox'][0]) * 100) / scan_width
                        h = ((article['bbox'][3] - article['bbox'][1]) * 100) / scan_height

                        # Create label studio sample
                        ls_item = [{
                            "data": {
                                "image": f"/data/local-files/?d=data/{article['image_file_name']}",
                                "headline": article['headline'],
                                "article": article['article'],
                                "byline": article['byline']
                            },

                            "predictions": [{
                                "result": [{
                                    "value": {
                                        "x": x,
                                        "y": y,
                                        "width": w,
                                        "height": h
                                    },
                                    "from_name": "label",
                                    "to_name": "image",
                                    "type": "rectanglelabels"
                                }]
                            }]
                        }]

                        for query in query_list:
                            if query in article["query"]:

                                os.makedirs(f'{save_dir}/{query}', exist_ok=True)
                                with open(f'{save_dir}/{query}/sample_{counter}.json', "w") as writer:
                                    writer.write(simplejson.dumps(ls_item, indent=4) + "\n")

                        counter += 1
