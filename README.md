# Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images

We present a model agnostic framework to inject prior
knowledge (PK) on gene-gene interactions into deep learning models predicting gene expression from WSIs. In our
research, we consider three different sources of PK, two feature extractors and three aggregators. On TCGA-BRCA, our
PK injection method led to an increase of 983 genes on average (across all 18 models), which transferred to an increase
in the CPTAC-BRCA dataset in 14 out of 18 cases. We conclude that injecting PK has potential to improve gene prediction performance and robustness across a wide range of
architectures

# Script templates for training, inference & gene embedding generation process

 'evaluate.sh': inference
 'train.sh': training
 'embeddings.sh': generating gene embeddings from an adjacency matrix of genes

