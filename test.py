from tqdm import tqdm
import glob
import ijson
import os
import simplejson

# path = '/mnt/data02/retrieval/retrieval_test/100_full_paper_samp_full/out_file_*'
path = 'C:/Users/Emily/Downloads/100_full_paper_samp_full/out_file_*'
# save_dir = '/mnt/data02/retrieval/retrieval_test/100_full_paper_samp_full_wids/'
save_dir = 'C:/Users/Emily/Downloads/100_full_paper_samp_full_wids/'

os.makedirs(save_dir, exist_ok=True)

for file_path in tqdm(glob.glob(path)):
    with open(file_path, 'rb') as f:
        id_file = {}
        items = ijson.kvitems(f, '')
        for k, v in items:
            v_with_ids = []
            for i in range(len(v)):
                ind_id = str(v[i]['full_article_id']) + '_' + v[i]['image_file_name']
                v[i]['id'] = ind_id
                v_with_ids.append(v[i])
            id_file[k] = v_with_ids

    save_end = file_path.split('\\')[-1:]                                                                               # might need changing for Guppy
    # save_end = file_path.split('/')[-1:]
    save_path = os.path.join(save_dir, save_end[0])

    with open(save_path, "w") as writer:
        writer.write(simplejson.dumps(id_file, indent=4) + "\n")