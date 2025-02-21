import re
import nltk
import spacy
import unicodedata
from nltk import TweetTokenizer
from spacy.lang.es import Spanish
from spacy.lang.en import English
from nltk.util import ngrams


class TextProcessing(object):
    name = 'Text Processing'
    lang = 'es'

    def __init__(self, lang: str = 'es'):
        self.lang = lang
        self.nlp = TextProcessing.load_spacy(lang=lang)

    @staticmethod
    def load_spacy(lang: str):
        result = None
        try:
            if lang == 'es':
                result = spacy.load('es_core_news_sm')
            else:
                result = spacy.load('en_core_web_sm')
            print('Language: {0}\n{1}: {2}'.format(TextProcessing.name, lang, result.pipe_names))
        except Exception as e:
            print('Error load_sapcy: {0}'.format(e))
        return result

    def analysis_pipe(self, text: str):
        doc = None
        try:
            doc = self.nlp(text=text)
        except Exception as e:
            print('Error analysis_pipe: {0}'.format(e))
        return doc

    @staticmethod
    def proper_encoding(text: str):
        result = ''
        try:
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            result = text.decode("utf-8")
        except Exception as e:
            print('Error proper_encoding: {0}'.format(e))
        return result

    @staticmethod
    def stopwords(text: str):
        result = ''
        try:
            nlp = Spanish()if TextProcessing == 'es' else English()
            doc = nlp(text)
            token_list = [token.text for token in doc]
            sentence = []
            for word in token_list:
                lexeme = nlp.vocab[word]
                if not lexeme.is_stop:
                    sentence.append(word)
            result = ' '.join(sentence)
        except Exception as e:
            print('Error stopwords: {0}'.format(e))
        return result

    @staticmethod
    def remove_patterns(text: str):
        result = ''
        try:
            text = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\#|\$|\€|\Â|\�|\¬', '', text)
            text = re.sub(r'\,|\;|\:|\!|\¡|\’|\‘|\”|\“|\"|\'|\`', '', text)
            text = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', '', text)
            text = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$', '', text)
            text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
            result = text.lower()
        except Exception as e:
            print('Error remove_patterns: {0}'.format(e))
        return result

    @staticmethod
    def transformer(text: str, stopwords: bool = False):
        result = ''
        try:
            text_out = TextProcessing.proper_encoding(text)
            text_out = text_out.lower()
            text_out = re.sub("[\U0001f000-\U000e007f]", '[EMOJI]', text_out)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '[URL]', text_out)
            text_out = re.sub("@([A-Za-z0-9_]{1,40})", '[MENTION]', text_out)
            text_out = re.sub("#([A-Za-z0-9_]{1,40})", '[HASTAG]', text_out)
            text_out = TextProcessing.remove_patterns(text_out)
            # text_out = TextAnalysis.lemmatization(text_out) if lemmatizer else text_out
            text_out = TextProcessing.stopwords(text_out) if stopwords else text_out
            text_out = re.sub(r'\s+', ' ', text_out).strip()
            text_out = text_out.rstrip()
            result = text_out if text_out != ' ' else None
        except Exception as e:
            print('Error transformer: {0}'.format(e))
        return result

    @staticmethod
    def tokenizer(text: str):
        val = []
        try:
            text_tokenizer = TweetTokenizer()
            val = text_tokenizer.tokenize(text)
        except Exception as e:
            print('Error make_ngrams: {0}'.format(e))
        return val

    @staticmethod
    def make_ngrams(text: str, num: int):
        result = ''
        try:
            n_grams = ngrams(nltk.word_tokenize(text), num)
            result = [' '.join(grams) for grams in n_grams]
        except Exception as e:
            print('Error make_ngrams: {0}'.format(e))
        return result

    @staticmethod
    def tagger(text: str):
        result = None
        try:
            list_tagger = []
            doc = TextProcessing.analysis_pipe(text=text)
            for token in doc:
                item = {'text': token.text, 'lemma': token.lemma_, 'stem': token._.stem, 'pos': token.pos_,
                        'tag': token.tag_, 'dep': token.dep_, 'shape': token.shape_, 'is_alpha': token.is_alpha,
                        'is_stop': token.is_stop, 'is_digit': token.is_digit, 'is_punct': token.is_punct}
                list_tagger.append(item)
            result = list_tagger
        except Exception as e:
            print('Error tagger: {0}'.format(e))
        return result