# Subframe Prediction

This repository contains dataset and codes for predicting subframes in text. The approach is describe in the following paper.

> [Weakly Supervised Learning of Nuanced Frames for Analyzing Polarization in News Media\
> Shamik Roy and Dan Goldwasser\
> Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.620.pdf)

## Dataset

The datasets can be found in the following Links -
1. [Abortion](https://drive.google.com/drive/folders/1g9P6vHTJ1nNec1zrBZggcNYu_Lmm95Oa?usp=sharing)
2. [Immigration](https://drive.google.com/drive/folders/1OAND83Jtng46WuVKMZx0dpO2o3UtbuUD?usp=sharing)
3. [Gun Control](https://drive.google.com/drive/folders/1nJ-kzZeIJkUzwRTjFAOAPTbMpd64N7L_?usp=sharing)
4. [General Frame Lexicon](https://drive.google.com/drive/folders/167EIDDD_spjdDHhewj_QVZiRv7FGAHL5?usp=sharing)

### Folders in 1, 2, 3 contains the following files -
1. **article_data.pkl**
2. **frame_data.pkl**
3. **subframe_data.pkl**

These files can be read by the procedure written in _'read_data.py'_. The description of the data-structures contained in these files are as follows.

1. **article_data.pkl**
* article2URL: type: dictionary; key: article; value: URL
* article2dop: type: dictionary; key: article; value: date of publication
* article2headline: type: dictionary; key: article; value: headline of the article
* article2text: type: dictionary; key: article; value: original text of the article
* article2label: type: dictionary; key: article; value: label of the article (left/right)
* article2segment_ids: type: dictionary; key: article; value: ids of the segments containing the article (described as paragraphs in the paper)
* seg_id2text: type: dictionary; key: segment ids; value: text of the segment
2. **frame_data.pkl**
* frame2bigrams: type: dictionary; key: frame; value: stemmed bigram lexicon for the frame
* frame2trigrams: type: dictionary; key: frame; value: stemmed trigram lexicon for the frame
* frame2sfs: type: dictionary; key: frame; value: subframes for the frame
3. **subframe_data.pkl**
* subframe2ngrams: type: dictionary; key: subframe; value: set of annotated stemmed bigrams and trigrams for the subframe

### Folder in 4 contains the general frame lexicon. It contains the following file - 
1. **general_frame_data.pkl**

These file can be read by the procedure written in _'read_data.py'_. The description of the data-structures contained in this file are as follows.

1. **general_frame_data.pkl**
* frame_name2lexicon: type: dictionary; key: frame; value: general stemmed unigram lexicon for the frame
* stem2words: type: dictionary; key: stem; value: corresponding set of words

### Additional Notes
1. All unigram, bigram and trigram lexicons are stemmed lexicons. **_NLTK Lancaster Stemmer_** was used for stemming in this project.
2. The un-stemmed phrases from the bigram and trigram lexicons can be found in the appendix of the paper.


## Embedding Learning Codes
### Required Pre-Processed Files: 
The following pre-processed files are needed to run the embedding learning.
1. **graph_info.pkl**
2. **tokenized_paragraphs.pkl**

These files for the topics _Abortion_, _Immigration_ and _Gun Control_ are contained in the folders containing the datasets for these three topics respectively. Their parsing information can be found in _'read_data.py'_.

### Implementation
The following three files contain the implementation of the embedding learning proposed in the paper.
1. **run.py**
2. **Embedder.py**
3. **BLSTM.py**

### Command 
`python run.py [-h] -i INPUT_FOLDER -o OUTPUT_FOLDER`
* `INPUT_FOLDER` : Path to the folder containing _graph_info.pkl_ and _tokenized_paragraphs.pkl_.
* `OUTPUT_FOLDER` : Path to the folder where the learned embeddings will be saved.

### Additional Notes and Acknowledgments
1. The learned embeddings were used for the qualitative and quantitative analysis.
2. GPUs will be needed to run the code in a reasonable amount of time.
3. The embedding learning codes are inspired from https://github.com/BillMcGrady/StancePrediction/tree/master/code. 

## Citation
If you find the approach helpful in your work, please cite the paper.

```
@inproceedings{roy2020weakly,
  title={Weakly Supervised Learning of Nuanced Frames for Analyzing Polarization in News Media},
  author={Roy, Shamik and Goldwasser, Dan},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={7698--7716},
  year={2020}
}
```

