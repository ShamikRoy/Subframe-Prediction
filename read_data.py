import pickle

def Read_Files():
    with open("article_data.pkl", "rb") as in_file:
        [article2URL, article2dop, article2headline, article2text, art2label, art2segment_ids, seg_id2text] = pickle.load(in_file)

    '''
    article2URL: type: dictionary; key: article; value: URL
    article2dop: type: dictionary; key: article; value: date of publication
    article2headline: type: dictionary; key: article; value: headline of the article
    article2headline: type: dictionary; key: article; value: original text of the article
    article2label: type: dictionary; key: article; value: label of the article (left/right)
    article2segment_ids: type: dictionary; key: article; value: ids of the segments containing the article (described as paragraphs in the paper)
    seg_id2text: type: dictionary; key: segment ids; value: text of the segment
    '''

    with open("frame_data.pkl", "rb") as in_file:
        [frame2bigrams, frame2trigrams, frame2sfs] = pickle.load(in_file)

    '''
    frame2bigrams: type: dictionary; key: frame; value: stemmed bigram lexicon for the frame
    frame2trigrams: type: dictionary; key: frame; value: stemmed trigram lexicon for the frame
    frame2sfs: type: dictionary; key: frame; value: subframes for the frame
    '''

    with open("subframe_data.pkl", "rb") as in_file:
        [subframe2ngrams] = pickle.load(in_file)

    '''
    subframe2ngrams: type: dictionary; key: subframe; value: set of annotated stemmed bigrams and trigrams for the subframe
    '''

    with open("general_frame_data.pkl", "rb") as in_file:
        [frame_name2lexicon, stem2word] = pickle.load(in_file)

    '''
    frame_name2lexicon: type: dictionary; key: frame; value: general stemmed unigram lexicon for the frame
    stem2words: type: dictionary; key: stem; value: corresponding set of words
    '''

    with open("graph_info.pkl", "rb") as in_file:
        [graph, id2name, name2id, doc_start, doc_end, \
         bigram_start, bigram_end, trigram_start, trigram_end, \
         subframe_start, subframe_end] = pickle.load(in_file)

    '''
    This file contains pre-processed format of the dataset suitable for the embedding learning.
    Every element - paragraph (doc), subframe, bigram, trigram; are encoded with a unique numeric id.
    The mapping to the encoded id and the original name of the element can be found in dictionaries named 
    'id2name' and 'name2id'.

    Then all of the elements are encoded to a graph structure. Where there exists an edge from a paragraph 
    (doc) to a bi/tri-gram if the bi/tri-gram is present in the paragraph, and a bi/tri-gram is connected to 
    its corresponding subframe label (if there is a label). This adjacency list can be found in 'graph'.

    doc_start, doc_end = starting and ending ids of the paragraphs
    bigram_srart, bigram_end = starting and ending ids of the bigrams
    trigram_srart, trigram_end = starting and ending ids of the trigrams
    subframe_srart, subframe_end = starting and ending ids of the subframes
    '''

    with open("tokenized_paragraphs.pkl", "rb") as in_file:
        [weights_matrix, segment2tokenized_text] = pickle.load(in_file)

    '''
    This file contains preprocessed data-structures containing segments (paragraphs) in a tokenized format (segment2tokenized_text).
    weight_matrix is a n*m dimensional matrix, where n=number of unique tokens in the whole corpus and m=GloVe embedding dimension (300 here).
    '''


if __name__ == '__main__':
    Read_Files()
