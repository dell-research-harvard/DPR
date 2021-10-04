# Python 3.7.11


"""
 Pipeline to take a random subsample of newspapers, by newspaper name, as a test set for inference.
"""

import random
import glob
from tqdm import tqdm
import ijson
import simplejson


def get_paper_name(file_end):
    return "-".join(file_end.split("-")[1:-5])


n_random_papers = 50
path_pattern = '/mnt/data02/retrieval/preprocess/all_scans_dbx_text_files_1970_full/**/ocr_*'

# Create list of file names
scan_names = []
for file_path in tqdm(glob.glob(path_pattern)):
    with open(file_path, 'rb') as f:
        items = ijson.kvitems(f, '')
        for k, v in items:
            scan_names.append(k)

papers = list(set([get_paper_name(scan) for scan in scan_names]))
print(f"{len(papers)} total papers...")

random_papers = random.sample(papers, n_random_papers)
print(f"Selected random papers: {random_papers}")

for paper in random_papers:
    random_selection = {}
    for file_path in tqdm(glob.glob(path_pattern)):
        with open(file_path, 'rb') as f:
            items = ijson.kvitems(f, '')
            for k, v in items:
                if get_paper_name(k) == paper:
                    # print({k:v})
                    random_selection[k] = v

    with open(f"/mnt/data02/retrieval/retrieval_test/100_full_paper_samp_full_BS/out_file_{paper}_70.json", "w") as writer:
        writer.write(simplejson.dumps(random_selection, indent=4) + "\n")