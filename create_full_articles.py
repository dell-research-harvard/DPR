# Python 3.7.11

"""
 Pipeline to group scrapes into full articles.
"""

import glob
from tqdm import tqdm
import ijson
import simplejson

from dpr.utils.qualitycheckingfunctions import *

path_pattern = '/mnt/data02/retrieval/preprocess/all_scans_dbx_text_files_1968_faro_lr4/**/ocr_*'
save_dir = '/mnt/data02/retrieval/preprocess/all_scans_dbx_text_files_1968_full_210929/'
spell_dict = load_lowercase_spell_dict('/mnt/data02/retrieval/preprocess/hunspell_word_dict_lower')

#path_pattern = 'C:/Users/Emily/Downloads/all_scans_dbx_text_files_1968_faro/**/ocr_*'
#save_dir = 'C:/Users/Emily/Downloads/all_scans_dbx_text_files_1968_full/'
#spell_dict = load_lowercase_spell_dict('C:/Users/Emily/Downloads/hunspell_word_dict_lower')


def create_full_articles(file_path):
    with open(file_path, 'rb') as f:
        fa_file = {}
        items = ijson.kvitems(f, '')
        for k, v in items:
            fa_v = []

            # Keep article and headline lobjs with full_article_ids and non-word rate < 100%
            clean_v = []
            fa_ids = []
            for i in range(len(v)):
                nwr = get_prop_non_words(v[i]["ocr_text"], spell_dict)
                if (v[i]["label"] == "article" or v[i]["label"] == "headline") and v[i]["full_article_id"] is not None and nwr < 1:
                    clean_v.append(v[i])
                    fa_ids.append(v[i]["full_article_id"])

            # Group by full_article_id
            for fa_id in set(fa_ids):
                fa_list = []
                ro_id = []
                for j in range(len(clean_v)):
                    if clean_v[j]["full_article_id"] == fa_id:
                        fa_list.append(clean_v[j])
                        ro_id.append(clean_v[j]["reading_order_id"])
                assert len(ro_id) == len(set(ro_id))

                # Merge into full article
                fa = {
                    "image_file_name": "",   # Should be the same for everything within FA
                    "image_path": "",        # Should be the same for everything within FA
                    "object_id": [],         # List of IDs for each object
                    "headline": "",          # From label and ocr_text
                    "article": "",           # From label and ocr_text
                    "bbox": [],              # Should be able to max/min the existing bounding boxes
                    "full_article_id": "",   # Should be the same for everything within FA
                    "id": "",                # Combination of full_article_id and image_file_name to create a unique id
                }

                for ro_id in set(ro_id):
                    for m in range(len(fa_list)):
                        if fa_list[m]["reading_order_id"] == ro_id:

                            if fa["image_file_name"] == "":
                                fa["image_file_name"] = fa_list[m]["image_file_name"]
                            else:
                                assert fa["image_file_name"] == fa_list[m]["image_file_name"]

                            if fa["image_path"] == "":
                                fa["image_path"] = fa_list[m]["image_path"]
                            else:
                                assert fa["image_path"] == fa_list[m]["image_path"]

                            fa["object_id"].append(fa_list[m]["object_id"])

                            if fa_list[m]["label"] == "headline":
                                fa["headline"] = fa["headline"] + fa_list[m]["ocr_text"] + " "

                            if fa_list[m]["label"] == "article":
                                fa["article"] = fa["article"] + fa_list[m]["ocr_text"] + " "

                            fa["bbox"].append(fa_list[m]["bbox"])

                            if fa["full_article_id"] == "":
                                fa["full_article_id"] = fa_list[m]["full_article_id"]
                            else:
                                assert fa["full_article_id"] == fa_list[m]["full_article_id"]

                            if fa["id"] == "":
                                fa["id"] = str(fa_list[m]['full_article_id']) + '_' + fa_list[m]['image_file_name']
                            else:
                                assert fa["id"] == str(fa_list[m]['full_article_id']) + '_' + fa_list[m]['image_file_name']

                # Reject articles with empty or very short length
                if len(fa["article"]) > 25:
                    fa_v.append(fa)

            if len(fa_v) > 0:
                fa_file[k] = fa_v

    return fa_file


if __name__ == '__main__':

    for path in tqdm(glob.glob(path_pattern)):
        full_article_file = create_full_articles(path)

        save_end = path.split('/')[-2:]                                                                           # might need changing for Guppy
        save_batch_dir = os.path.join(save_dir, save_end[0])
        os.makedirs(save_batch_dir, exist_ok=True)

        save_path = os.path.join(save_dir, save_end[0], save_end[1])
        with open(save_path, "w") as writer:
            writer.write(simplejson.dumps(full_article_file, indent=4) + "\n")

