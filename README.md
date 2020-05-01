# ProQA

Resource-efficient method for pretraining a dense corpus index for open-domain QA and IR

## Requirements
* ``Python 3``
* ``Pytorch 1.4``
* ``tensorboardX``
* ``transformers``
* ``faiss``

## Retriever pretraining
### Pretraining with a single file:
* Pretraining under the ``retrieval'' directory: ``sh train_retriever.sh``

### Pretraining with clusters:
#### Generate clusters
* Generate the paragraph embeddings: ``sh get_para_embed.sh``
* Generate clusters using the paragraph embeddings: ``python group_paras.py /path/of/paragraph/embeddings /path/to/save/clusters``

#### Pretraining using clusters
* Change the training path in train_retriever.sh to ``/path/of/folder/clutering/dada``
* Then run the retrieval script: ``sh train_retriever.sh``

## QA finetuning
* Finetune the pretraining model on the QA dataset: ``train_dense_qa.sh``
