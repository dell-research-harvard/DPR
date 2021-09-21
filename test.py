import ijson
import collections


def get_paper_name(file_end):
    return "-".join(file_end.split("-")[1:-5])


def ocr_text_iter(v):
    for ik in v:
        if ik['label'] == "article":
            if page_filter:
                if not ik['image_file_name'].split('.')[0].endswith(f'p-{self.page_filter}'):
                    yield (ik['image_file_name'], ik['ocr_text'], ik['object_id'])
            else:
                yield (ik['image_file_name'], ik['ocr_text'], ik['object_id'])


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("-\n", "")
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    ctx_text = ctx_text.encode('ascii', 'ignore').decode()
    ctx_text = ctx_text.translate(str.maketrans('', '', r'"#$%&\()*+/:;<=>@[\\]^_`{|}~'))
    ctx_text = ctx_text.strip()
    return ctx_text


BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

n_random_papers = False
random_papers = []
file_path = "C:/Users/Emily/Downloads/out_file_caribou-county-sun.json"
page_filter = None
normalize = True
layout_object = 'article'


with open(file_path, 'rb') as f:
    items = ijson.kvitems(f, '')
    ocr_text_generators = []
    for k, v in items:
        if n_random_papers:
            if get_paper_name(k) in random_papers:
                ocr_text_generators.append(ocr_text_iter(v))
        else:
            ocr_text_generators.append(ocr_text_iter(v))

print(len(ocr_text_generators))

for gen in ocr_text_generators:
    for layobj in gen:
        title, passage, object_id = layobj
        uid = str(object_id) + '_' + title
        if normalize:
            if layout_object == 'headline':
                passage = normalize_passage(passage)
                passage = passage.lower()
            else:
                passage = normalize_passage(passage)
        print(BiEncoderPassage(passage, title))
