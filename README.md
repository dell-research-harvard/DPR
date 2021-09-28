**d**# Dense Passage Retrieval

Dense Passage Retrieval (DPR) - was originally a set of tools and models for Q&A research.
- Based on the following paper: Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Preprint 2020.
- The [original repo](https://github.com/facebookresearch/DPR) includes comprehensive documentation. 

The DPR model has two core modules: a "retriever" and a "reader." In a question-answering paradigm, for which DPR was originally intended, the retriever is responsible for retrieving content (e.g., passages from a website, articles) that is likely to contain an answer to some query/question, and the reader is responsible for reading the retrieved content to extract a precise answer to said question. In other words, the output of a DPR retriever is a ranked set of raw passages from your corpus/dataset, where a higher rank corresponds to a passage that is more likely to contain an answer to your query, and the output of a DPR reader is usually just a sequence of a few words from each of those passages that represents the exact answer to your question.

DPR consists of four main tools:
1. Retriever training (train_dense_encoder.py)
2. Retriever inference on contexts. This embeds all articles where the answer might be found. (generate_dense_embeddings.py)
3. Retriever inference on questions. This embeds the question(s) and compares with the context embeddings to give a similarity score and output the most similar contexts (dense_retriever.py). 
4. Reader model training and inference - not used (train_reader.py)

We have adapted this model for topic retrieval, so that instead of questions, the retriever model is fed a specific topic and scores contexts on how likely to they are to be about this topic. To this end, we are not particularly interested in the reader module and therefore focus on the first three tools. The below addresses each of these tools in turn. And end-to-end example is then given. The authors of the oriignal DPR model provide an end-to-end example in their README, which may be useful. 


## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone git@github.com:dell-research-harvard/DPR.git
cd DPR
pip install .
```

## Tools 

### 1. Retriever training
Retriever training quality depends on its effective batch size. The one reported in the paper used 8 x 32GB GPUs.
In order to start training on one machine:
```bash
python train_dense_encoder.py \
train_datasets=[list of train datasets, comma separated without spaces] \
dev_datasets=[list of dev datasets, comma separated without spaces] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

Example for NQ dataset

```bash
python train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

DPR uses HuggingFace BERT-base as the encoder by default. Other ready options include Fairseq's ROBERTA and Pytext BERT models.
One can select them by either changing encoder configuration files (conf/encoder/hf_bert.yaml) or providing a new configuration file in conf/encoder dir and enabling it with encoder={new file name} command line parameter. 

Notes:
- If you want to use pytext bert or fairseq roberta, you will need to download pre-trained weights and specify encoder.pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- Validation and checkpoint saving happens according to train.eval_per_epoch parameter value.
- There is no stop condition besides a specified amount of epochs to train (train.num_train_epochs configuration parameter).
- Every evaluation saves a model checkpoint.
- The best checkpoint is logged in the train process output.
- Regular NLL classification loss validation for bi-encoder training can be replaced with average rank evaluation. It aggregates passage and question vectors from the input data passages pools, does large similarity matrix calculation for those representations and then averages the rank of the gold passage for each question. We found this metric more correlating with the final retrieval performance vs nll classification loss. Note however that this average rank validation works differently in DistributedDataParallel vs DataParallel PyTorch modes. See train.val_av_rank_* set of parameters to enable this mode and modify its settings.

See the section 'Best hyperparameter settings' below as e2e example for our best setups.

#### Distributed training
Use Pytorch's distributed training launcher tool:

```bash
python -m torch.distributed.launch \
	--nproc_per_node={WORLD_SIZE}  {non distributed scipt name & parameters}
```
Note:
- all batch size related parameters are specified per gpu in distributed mode(DistributedDataParallel) and for all available gpus in DataParallel (single node - multi gpu) mode.



### 2. Embedding contexts offline for retrieval 

DPR introduces large efficiency gains by only embedding each context once. These are then stored offline to be compared with the topic embedding during retrieval. 

```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```

- `model_file`: path to the best checkpoint.
- `ctx_scr`: choice of resources from ctx_sources config file (eg. `dpr_wiki`, `newspaper_archive`).
- `shard_id`: Generating representation vectors for the static contexts dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.
- `outfile`: path to results folder

Note: you can use much large batch size here compared to training mode. For example, setting batch_size 128 for 2 GPU(16gb) server should work fine.


### 3. Embedding the topic, comparing to contexts 

With embeddings now created "offline" for a universe of articles of interest, we now create an embedding for some topic, and then compare that topic embedding to all the context embeddings (by computing their dot product). 

```bash
python dense_retriever.py \
	model_file={path to biencoder checkpoint} \
	qa_dataset={the name of the test source} \
	ctx_datatsets=[topic source name] \
	encoded_ctx_files=[filepath for outfile from embedding contexts offline] \
	out_file={path to output json file with results} 	
```

- `model_file`: path to the best checkpoint, same as when embedding contexts offline.
- `qa_dataset`: choice of questions/topics from retriever_default.yaml. `custom_test` can be used for topic retrieval, with the `question:` field amended as appropriate.
- `ctx_dataset`: choice of resources from ctx_sources config file (eg. `dpr_wiki`, `newspaper_archive`). Same as `ctx_scr` when embedding contexts offline. This is used to match retrieval scores back to the inputs to create the output file. 
- `encoded_ctx_files`: path to output from embedding contexts offline. Note that the output file specified in `out_file` when embedding contexts offline gives the prefix and therefore the filepath here will be marginally different. 
- `outfile`: path to output json file with results.


For example: 

```bash
python dense_retriever.py \
	model_file={path to a checkpoint file} \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=[\"~/myproject/embeddings_passages1/wiki_passages_*\",\"~/myproject/embeddings_passages2/wiki_passages_*\"] \
	out_file={path to output json file with results} 
```


The tool writes retrieved results into specified out_file as a json with the following format:

```
[
    {
        "question": "...",
        "answers": ["...", "...", ... ],
        "ctxs": [
            {
                "id": "...", # passage id from database tsv file
                "title": "",
                "text": "....",
                "score": "...",  # retriever score
                "has_answer": true|false
     },
]
```
Results are sorted by their similarity score, from most relevant to least relevant.

By default, the retriever model will output a score for all context passages. To change this, the `n_docs:` parameter can be changed in the dense_retriever.yaml config file, or in the command line. 


### 4. Reader model 
The reader model was not used. For information on reader model training and inference, see the [original DPR repo](https://github.com/facebookresearch/DPR).



## Using DPR on your own data. 
First, you need to prepare data for retriever training.
Each of the DPR components has its own input/output data formats.
You can see format descriptions below.

### Retriever input data format
The default data format of the Retriever training data is JSON.
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```

Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs.
The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.


### DPR data formats and custom processing
One can use their own data format and custom data parsing & loading logic by inherting from DPR's Dataset classes in dpr/data/{biencoder|retriever|reader}_data.py files and implementing load_data() and __getitem__() methods. See [DPR hydra configuration](https://github.com/facebookresearch/DPR/blob/master/conf/README.md) instructions.



