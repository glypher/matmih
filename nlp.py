"""nlp.py: Classes for working with NLP data
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import nltk
import numpy as np
import unidecode


class PreprocessPipeline:

    def __init__(self, df, language):
        self._df = df
        self._language = language

    def _split_dataframe(self, functor):
        newDF = pd.concat([pd.Series(row['sid'], functor(row['text']))
                           for _, row in self._df.iterrows()]).reset_index()
        newDF = newDF.rename(columns={'index': "text", 0: "sid"})
        newDF = newDF.merge(self._df[['target', 'sid']], on="sid", how='inner')

        return newDF

    def split_sentences(self):
        self._df = self._split_dataframe(nltk.sent_tokenize)
        return self

    def lower(self):
        self._df['text'] = self._df['text'].apply(lambda s: s.lower())
        return self

    def tokenize(self):
        self._word_list = True
        self._df['text'] = self._df['text'].apply(lambda s: nltk.word_tokenize(s))
        return self

    def stem(self):
        stemmer = nltk.SnowballStemmer(self._language)
        self._df['text'] = self._df['text'].apply(lambda s: [stemmer.stem(w) for w in s])
        return self

    def remove_punctuation(self):
        self._df['text'] = self._df['text'].apply(lambda s: [w for w in s if w.isalnum()])
        return self

    def remove_diacritics(self):
        self._df['text'] = self._df['text'].apply(unidecode.unidecode)
        return self

    def remove_stopwords(self):
        self._df['text'] = self._df['text'].apply(
            lambda s: [w for w in s if w not in nltk.corpus.stopwords.words(self._language)])
        return self

    def convert_to_phonames(self):
        arpabet = nltk.corpus.cmudict.dict()
        self._df['text'] = self._df['text'].apply(lambda s: [arpabet[w][0] for w in s])

    def build_vocabulary(self, vocab: dict):
        for _, row in self._df.iterrows():
            for w in row['text']:
                if w not in vocab:
                    vocab[w] = len(vocab) + 1
        return self

    def to_vocabulary_ids(self, vocab, default_value=0):
        self._df['text'] = self._df['text'].apply(lambda s: np.array([vocab.get(w, default_value) for w in s], dtype=np.int))
        return self

    def join_words(self):
        self._df['text'] = self._df['text'].apply(lambda s: ''.join([w + ' ' if w.isalnum() else w for w in s]))
        return self

    @property
    def DF(self):
        return self._df

    def process(self, pipeline: list):
        preprocess = self
        for func_name in pipeline:
            func = getattr(PreprocessPipeline, func_name)
            preprocess = func(preprocess)

        return self