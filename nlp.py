"""nlp.py: Classes for working with NLP data
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import nltk
import numpy as np
import unidecode
import pandas as pd


class PreprocessPipeline:
    CACHE = {}

    def __init__(self, df, language, vocab={}, copy=True, log=False, custom_split=None, min_words=1, max_words=128, min_word_count=5):
        self._df = df
        self._vocab = vocab
        self._log = log
        self._custom_split = custom_split
        self._min_words = min_words
        self._max_words = max_words
        self._min_word_count = min_word_count
        self._id = f"{type(self._df)}_{id(self._df)}_{min_words}_{max_words}_{min_word_count}"
        if copy:
            self._df = self._df.copy()
        self._language = language

    def _split_dataframe(self, functor):
        newDF = pd.concat([pd.Series(row['sid'], functor(row['text']))
                           for _, row in self._df.iterrows()]).reset_index()
        newDF = newDF.rename(columns={'index': "text", 0: "sid"})
        newDF = newDF.merge(self._df[['target', 'sid']], on="sid", how='inner')

        return newDF

    def split_sentences(self):
        if self._custom_split is None:
            self._df = self._split_dataframe(nltk.sent_tokenize)
        else:
            def _tokenize(s):
                s = nltk.sent_tokenize(s)
                s = [split for s1 in s for split in s1.split(self._custom_split)]
                return s
            self._df = self._split_dataframe(_tokenize)
        return self

    def split_max_word_sentences(self):
        def _chunks(s):
            return [' '.join(s[i : i+self._max_words]) for i in range(0, len(s), self._max_words)]
        self._df = self._split_dataframe(_chunks)
        self._df['text'] = self._df['text'].apply(lambda s: s.split(' '))
        return self

    def lower(self):
        self._df['text'] = self._df['text'].apply(lambda s: s.lower())
        return self

    def tokenize(self):
        self._df['text'] = self._df['text'].apply(lambda s: nltk.word_tokenize(s))
        return self

    def length(self):
        self._df['text'] = self._df['text'].apply(lambda s: [len(w) for w in s])
        return self

    def stem(self):
        stemmer = nltk.SnowballStemmer(self._language)
        self._df['text'] = self._df['text'].apply(lambda s: [stemmer.stem(w) for w in s])
        return self

    def pos_tag(self):
        self._df['text'] = self._df['text'].apply(lambda s: [p for w, p in nltk.pos_tag(s)])
        return self

    def remove_punctuation(self):
        self._df['text'] = self._df['text'].apply(lambda s: [w for w in s if not w.isalnum()])
        return self

    def remove_diacritics(self):
        self._df['text'] = self._df['text'].apply(unidecode.unidecode)
        return self

    def remove_stopwords(self):
        stopwords = nltk.corpus.stopwords.words(self._language)
        self._df['text'] = self._df['text'].apply(
            lambda s: [w for w in s if w not in stopwords])
        return self

    def only_stopwords(self):
        stopwords = nltk.corpus.stopwords.words(self._language)
        self._df['text'] = self._df['text'].apply(
            lambda s: [w for w in s if w in stopwords])
        return self

    def convert_to_phonames(self):
        arpabet = nltk.corpus.cmudict.dict()
        # Vowel lexical stress in cmudict: 0 — No stress,  1 — Primary stress, 2 — Secondary stress
        self._df['text'] = self._df['text'].apply(lambda s: [arpabet[w][0] for w in s if w in arpabet])
        self._df['text'] = self._df['text'].apply(lambda s: [w for words in s for w in words])
        return self

    def build_vocabulary(self):
        if len(self._vocab) > 0:
            return self

        vocab_count = {}
        for _, row in self._df.iterrows():
            for w in row['text']:
                if w not in vocab_count:
                    vocab_count[w] = 1
                else:
                    vocab_count[w] += 1

        for w, count in vocab_count.items():
            if count > self._min_word_count:
                self._vocab[w] = len(self._vocab) + 1

        return self

    def to_vocabulary_ids(self, default_value=0):
        self._df['text'] = self._df['text'].apply(lambda s: np.array([self._vocab.get(w, default_value) for w in s], dtype=np.int))
        return self

    def filter_rows(self):
        self._df['text'] = self._df['text'].apply(lambda s: pd.NA if len(s) < self._min_words else s)
        self._df = self._df.dropna().reset_index()
        return self

    def remove_pad_ids(self, default_value=0):
        self._df['text'] = self._df['text'].apply(lambda s: np.array([w for w in s if w != default_value], dtype=np.int))
        return self

    def join_words(self):
        self._df['text'] = self._df['text'].apply(lambda s: ''.join([str(w) + ' ' for w in s]))
        return self

    @property
    def DF(self):
        return self._df

    @property
    def VOCAB(self):
        return  self._vocab

    def _process(self, pipeline:list):
        preprocess = self
        for func_name in pipeline:
            func = getattr(PreprocessPipeline, func_name)
            preprocess = func(preprocess)
        return preprocess

    def process(self, pipeline: list):
        preprocess = self
        cache_ind = [i for i, op in enumerate(pipeline) if op == 'cache']
        done = []
        last_cid = 0
        for cid in cache_ind:
            to_do = pipeline[last_cid:cid]
            last_cid = cid+1
            done += to_do

            data_id = f"{self._id}_{'_'.join(done)}"
            if data_id in PreprocessPipeline.CACHE:
                if self._log:
                    print(f'Loading pipeline cached {data_id}...')
                preprocess = PreprocessPipeline.CACHE[data_id]
            else:
                preprocess = self._process(to_do)
                if self._log:
                    print(f'Saving to pipeline cache {data_id}...')
                PreprocessPipeline.CACHE[data_id] = preprocess

        preprocess = preprocess._process(pipeline[last_cid:])
        return preprocess
