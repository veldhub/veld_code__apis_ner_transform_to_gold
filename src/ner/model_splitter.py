import logging
import spacy
from spacy.attrs import ORTH

class Tokenizer:
    def __init__(self, pipes_to_disable=['tagger', 'parser', 'ner'], load_nlp=True):
        if load_nlp:
            self.nlp = spacy.load("de_core_news_md", disable=pipes_to_disable)
        self.logger = logging.getLogger(__name__)

    def add_abbreviations_from_file(self, abbreviations_file_path):
        self.abbreviations_file_path = abbreviations_file_path
        with open(abbreviations_file_path, "r") as file:
            abbreviated_words = [ x[:x.find('#')].strip() for x in file.readlines() if not x.startswith("#") ]
        for w in abbreviated_words:
            self.nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.logger.debug(f"Added {len(abbreviated_words)} abbreviations to tokenizer.")

    def tokenize(self, text):
        return self.nlp.make_doc(text)

class SentenceSplitter(Tokenizer):
    def __init__(self, pipes_to_disable=['ner'], load_nlp=True):
        super().__init__(pipes_to_disable=pipes_to_disable, load_nlp=load_nlp)

    def sentencize(self, text):
        doc = self.nlp.make_doc(text)
        doc = self.nlp.get_pipe('tagger')(doc)
        doc = self.nlp.get_pipe('parser')(doc)
        assert doc.is_sentenced
        sents = list(doc.sents)
        self.logger.debug(f"Found {len(sents)} sentences.")
        return doc

class NoSentenceSplitter(Tokenizer):
    """Pretends to be a SentenceSplitter, but doesn't split text into sentences. """

    def __init__(self, pipes_to_disable=['tagger', 'parser', 'ner'], load_nlp=True):
        super().__init__(pipes_to_disable=pipes_to_disable, load_nlp=load_nlp)

    def sentencize(self, text):
        """Doesn't actually sentencize. """
        doc = self.nlp(text)
        return doc
