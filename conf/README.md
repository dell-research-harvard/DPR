# Configuration


### Configuration for each tool

Each of the four main tools (train_dense_encoder.py, generate_dense_embeddings.py, dense_retriever.py and train_reader.py) take their parameters from a corresponding configuration file in this folder. 
* biencoder_train_cfg.yaml is the conf file for train_dense_encoder.py
* gen_embs.yaml is the conf file for generate_dense_embeddings.py
* dense_retriever.yaml is the conf file for dense_retriver.py
* extractive_reader_train_cfg.yaml is the conf file for train_extractive_reader.py

### Configuration Groups
The configuration files are organised using [Hydra](https://github.com/facebookresearch/hydra), which is an open-source Python
framework that allows a hierarchical configuration. 

In practice, this means that the main config file for each tool refers to the other config files (in the folders) for all the configuration parameters. These are called [configuration groups](https://hydra.cc/docs/tutorials/structured_config/config_groups) in Hydra. The main configuration files refers to other configuration files via the "defaults:" parameter.

For example in dense_retriever.yaml (which gives the configuration for dense_retriever.py) the `"defaults:"` parameter is as follows:

```yaml
defaults:
  - encoder: hf_bert
  - datasets: retriever_default
  - ctx_sources: default_sources
```
This tells dense_retriever.py that it should use hf_bert.yaml for the encoder configuration, retriever_default.yaml for dataset configuration etc. 

There are four configuration groups:
#### 1. `" - encoder:"` 
Specifies which model to be used for encoding (eg. BERT, RoBERTa etc.), as well as which parameters to use to instantiate the encoder.

#### 2. `" - train:"` 
(Used only for biencoder_train_cfg.yaml and extractive_reader_train_cfg.yaml ie. retriever and reader training) specifies training hyperparameters.  

#### 3.`" - datasets:"` 
(Used only for biencoder_train_cfg.yaml and dense_retriever.yaml) contains a list of all possible sources of hypotheses (ie. topics or queries in the original paper) that we are aiming to retrieve/evaluate. Each source is presented as a separate subsection of the config file.
For each source `_target_:` specifies which class to instantiate and `file_path` gives the location of the data.

One subsection of the specified config file is used at a time for inference (ie. one hypothesis source). This can be specified using the `qa_dataset` parameter when running dense_retriever.py from the command line. For example, if you want to run the retriever on NQ test set, set `qa_dataset=nq_test` as a command line parameter.

For training, multiple sources of hypotheses may be specified, using the `train_datasets` and `dev_datasets` parameters. 


#### 4. `" - ctx_sources:"` 
(Used only for gen_embs.yaml and dense_retriever.yaml ie. inference using the retriever) - contains a list of all possible passage sources ie. articles which are possible candidates for finding the specified topic/evaluating the specified hypothesis (equivalent to passages that may contain possible answers in the original paper). These are sometimes also referred to as contexts.

Similarly to above, the context source is specified in the command line, using `ctx_scr` for generate_dense_embeddings.py and `ctx_datatsets` when running dense_retriever.py. For example, if you want to use wikipedia passages as contexts, set `ctx_datatsets=[dpr_wiki]`.
This parameter is a list and you can effectively concatenate different passage source into one. In order to use multiple sources at once, one also needs to provide relevant embeddings files in `encoded_ctx_files` parameter in the command line, which is also a list.

To use custom datasets define your own class that provides relevant `__getitem__()`, `__len__()` and `load_data()` methods (inherit from QASrc). For example, we created the `NewspaperArchiveCtxSrc` class in retriever_data.py to be able to use a dataset of newspapers.


### Parameters 
Refer to the configuration files for comments for every parameter.

If you want to override parameters within these config files, there are multiple options: 
* Modify that config file directly.
* Create a new config group file in the relevant folder and use it by providing the `{config group}={your file name}` command line argument. For example, to modify the encoder parameters, create a new file in the conf/encoder/ folder and use it by providing encoder={your file name} command line argument. Create a new config group file in the relevant folder under  conf/encoder/ folder and enable to use it by providing `encoder={your file name}` command line argument.
* Override specific parameter from command line. For example: `encoder.sequence_length=300`.




