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

if __name__ == '__main__':
    Read_Files()
