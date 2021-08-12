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

    def __init__(self, df, language, vocab={}, copy=True, log=False, custom_split=None, min_words=1,
                 max_words=128, min_word_count=5, column_name='text', mask_column='attention_mask',
                 padding=128, padding_id=0, end_token_id=1):
        self._df = df
        self._vocab = vocab.copy()
        self._log = log
        self._custom_split = custom_split
        self._min_words = min_words
        self._max_words = max_words
        self._min_word_count = min_word_count
        self._column_name = column_name
        self._mask_column = mask_column
        self._padding = padding
        self._padding_id = padding_id
        self._end_token_id = end_token_id
        self._id = f"{type(self._df)}_{id(self._df)}_{min_words}_{max_words}_{min_word_count}_{column_name}_{mask_column}_{padding}_{padding_id}_{end_token_id}"
        assert end_token_id > padding_id, 'End token id > padding id'
        if copy:
            self._df = self._df.copy()
        self._language = language

    def _split_dataframe(self, functor):
        newDF = pd.concat([pd.Series(row['sid'], functor(row[self._column_name]))
                           for _, row in self._df.iterrows()]).reset_index()
        newDF = newDF.rename(columns={'index': self._column_name, 0: "sid"})
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
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: s.split(' '))
        return self

    def lower(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: s.lower())
        return self

    def tokenize(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: nltk.word_tokenize(s))
        return self

    def tokenize_char(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [c for c in s])
        return self

    def length(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [len(w) for w in s])
        return self

    def stem(self):
        stemmer = nltk.SnowballStemmer(self._language)
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [stemmer.stem(w) for w in s])
        return self

    def pos_tag(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [p for w, p in nltk.pos_tag(s)])
        return self

    def remove_punctuation(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [w for w in s if w.isalnum()])
        return self

    def remove_diacritics(self):
        self._df[self._column_name] = self._df[self._column_name].apply(unidecode.unidecode)
        return self

    def remove_stopwords(self):
        stopwords = nltk.corpus.stopwords.words(self._language)
        self._df[self._column_name] = self._df[self._column_name].apply(
            lambda s: [w for w in s if w not in stopwords])
        return self

    def only_stopwords(self):
        stopwords = nltk.corpus.stopwords.words(self._language)
        self._df[self._column_name] = self._df[self._column_name].apply(
            lambda s: [w for w in s if w in stopwords])
        return self

    def convert_to_phonames(self):
        arpabet = nltk.corpus.cmudict.dict()
        # Vowel lexical stress in cmudict: 0 — No stress,  1 — Primary stress, 2 — Secondary stress
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [arpabet[w][0] for w in s if w in arpabet])
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: [w for words in s for w in words])
        return self

    def build_vocabulary(self):
        vocab_count = {}
        for _, row in self._df.iterrows():
            for w in row[self._column_name]:
                if w not in vocab_count:
                    vocab_count[w] = 1
                else:
                    vocab_count[w] += 1

        self._vocab['<p>'] = self._padding_id
        self._vocab['<eos>'] = self._end_token_id
        ids = max(self._vocab.values())
        for w, count in vocab_count.items():
            if w not in self._vocab and count > self._min_word_count:
                ids += 1
                self._vocab[w] = ids

        return self

    def to_vocabulary_ids(self, default_value=0):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: np.array([self._vocab.get(w, default_value) for w in s], dtype=np.int))
        return self

    def add_mask(self):
        self._df[self._mask_column] = self._df[self._column_name].apply(lambda s: np.ones_like(s))
        return self

    def add_end_token(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: np.concatenate((s, [self._end_token_id])))
        if self._mask_column in self._df:
            self._df[self._mask_column] = self._df[self._mask_column].apply(lambda s: np.concatenate((s, [1])))
        return self

    def padding(self):
        self._df[self._column_name] = self._df[self._column_name].apply(
            lambda s: np.pad(s, (0, self._padding - len(s)), constant_values=self._padding_id)
            if len(s) < self._padding else np.resize(s, self._padding))
        if self._mask_column in self._df:
            self._df[self._mask_column] = self._df[self._mask_column].apply(
                lambda s: np.pad(s, (0, self._padding - len(s)), constant_values=0)
                if len(s) < self._padding else np.resize(s, self._padding))
        return self

    def filter_rows(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: pd.NA if len(s) < self._min_words else s)
        self._df = self._df.dropna().reset_index()
        return self

    def remove_pad_ids(self, default_value=0):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: np.array([w for w in s if w != default_value], dtype=np.int))
        return self

    def join_words(self):
        self._df[self._column_name] = self._df[self._column_name].apply(lambda s: ''.join([str(w) + ' ' for w in s]))
        return self

    @property
    def DF(self):
        return self._df

    @property
    def VOCAB(self):
        return self._vocab

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
