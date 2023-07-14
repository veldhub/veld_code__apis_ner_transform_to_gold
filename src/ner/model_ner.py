from .corpus_dummy import *
from .data import *
from .model_splitter import SentenceSplitter

from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
import pickle
import random
import spacy
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
import subprocess

class NERer(SentenceSplitter):

    # key of `map` = key for pickle dictionary
    # value of `map` = name of variable in class NERer
    map = {}
    map["model type"] = "model_type"
    map["training style"] = "training_style"
    map["training iterations"] = "training_iterations"
    map["saved as"] = "saved_as"
    map["corpus version"] = "corpus_name"
    map["data version"] = "data_version"
    map["eval scores"] = "scores"
    map["eval scores (manual)"] = "scores_manual"

    def __init__(self, corpus_name="Corpus", path_to_vectors=None,
    model_type=None, training_style=None, training_iterations=None,
    data=None, training_data=[], evaluation_data=[], load_nlp=True):

        super().__init__(load_nlp=load_nlp)
            # self.nlp will be set to the German model with tagger and parser pipes
            # the NER pipe will be set by self.train()

        self.corpus_name = corpus_name
        if self.corpus_name:
            self.corpus = import_class(f'ner.corpus_dummy.{self.corpus_name}')()
        else:
            self.corpus = Corpus()
        self.path_to_vectors = path_to_vectors

        self.model_type = model_type
        self.training_style = training_style
        self.training_iterations = training_iterations

        self.data = data
        self.training_data = training_data
        self.evaluation_data = evaluation_data

        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_saved(cls, path, load_training_data=True, load_evaluation_data=True):
        n = cls(load_nlp=False, corpus_name=None)
        n.logger.debug(f"Loading NERer from {path}.")
        n.load_metadata(f'{path}/model_dict.pickle')
        n.load_model(f'{path}/nlp')
        if load_training_data:
            n.load_training_data(f'{path}/corpus/trainset.pickle')
        if load_evaluation_data:
            n.load_evaluation_data(f'{path}/corpus/evalset.pickle')
        n.path_to_vectors = f'{path}/corpus/vectors'
        return n

    ############################################ LOAD DATA
    def load_metadata(self, path):
        self.logger.debug(f"Loading metadata from {path}.")
        dict = pickle.load(open(path, "rb"))
        for key in dict:
            if key in self.map:
                self.__dict__[self.map[key]] = dict[key]
        if self.corpus_name == 4:
            self.corpus_name = 'Corpus4'

    def load_model(self, path):
        self.logger.debug(f"Loading model from {path}.")
        self.nlp = spacy.load(path)

    def __load_data(self, path):
        data_without_goldparse = pickle.load(open(path, "rb"))
        # if the data is in the new format:
        if type(data_without_goldparse[0]) is Datapoint:
            self.logger.debug("We have loaded Datapoints.")
            data_with_goldparse = Dataset(data_without_goldparse)
            for d in data_with_goldparse:
                if d.doc == None:
                    d.sentencize(self)
                d.ents_into_doc()
                if d.goldparse == None:
                    d.parse_gold()
        # the data is in the old format
        elif type(data_without_goldparse[0]) is tuple:
            self.logger.debug("We have loaded tuples.")
            data_with_goldparse = Dataset()
            for d in data_without_goldparse:
                # sentence as string
                assert type(d[0]) is str
                sentence = d[0]
                # entities as list of NamedEntities
                if type(d[1]) is dict and "entities" in d[1]:
                    entities = [ NamedEntity.from_tuple(e) for e in d[1]["entities"] ]
                elif type(d[1]) is list and type(d[1][0]) is NamedEntity:
                    # if our corpus contains sentences without annotations,
                    # accessing d[1][0] might throw an error
                    entities = d[1]
                else:
                    assert False
                datapoint = Datapoint(sentence, entities)
                # doc
                if type(d[2]) is spacy.tokens.doc.Doc:
                    datapoint.doc = d[2]
                else:
                    datapoint.sentencize(self)
                datapoint.ents_into_doc()
                # GoldParse
                datapoint.parse_gold()
                data_with_goldparse.append(datapoint)
        else:
            assert False
        return data_with_goldparse

    def load_training_data(self, path):
        self.logger.debug(f"Loading training data from {path}.")
        self.training_data = self.__load_data(path)

    def load_evaluation_data(self, path):
        self.logger.debug(f"Loading evaluation data from {path}.")
        self.evaluation_data = self.__load_data(path)

    ############################################ TRAIN
    def train(self):

        self.logger.debug(f"Training style: {self.model_type} ...")
        assert self.nlp is not None
        self.nlp = self.model_type.create_ner_pipe(nlp=self.nlp)

        self.logger.debug(f"Training on labels {self.corpus.labels} ...")
        for label in self.corpus.labels:
            self.nlp.get_pipe("ner").add_label(label)

        self.logger.debug(f"Updating vectors from {self.path_to_vectors} ...")
        # TODO: this should be a point of variation:
        # do we use the vectors from the training+eval set,
        # or the vectors for the entire corpus that the model will be applied to?
        self.nlp.vocab.vectors.from_disk(self.path_to_vectors)
        self.logger.info(f"Vector dimensions: {self.nlp.vocab.vectors.shape}")

        for d in self.training_data:
            assert d.doc is not None
            d.ents_into_doc()
            if d.goldparse == None:
                d.parse_gold()

        with self.nlp.disable_pipes('tagger', 'parser'): # only train the NER pipe

            # if self.training_style == TrainingStyle.GOLD:
            #     self.logger.debug("Preprocessing training data ...")
            #     self.nlp.preprocess_gold([ (d.doc, d.goldparse) for d in self.training_data ])
            #         # To do: this returns tuples of Docs and GoldParses that should be used for training

            self.logger.debug("Initializing weights ...")
            optimizer = self.nlp.begin_training()

            for i in range(self.training_iterations):
                self.logger.debug(f"Iteration #{i} ...")
                random.shuffle(self.training_data)
                losses = {}
                batches = minibatch(self.training_data, size = compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations, docs, goldparses = zip(*batch)
                    if self.training_style == TrainingStyle.SIMPLE:
                        self.nlp.update(
                            texts,
                            tuple([ {"entities": [ e.to_tuple() for e in a ]} for a in annotations ]),
                            drop=0.5,
                            losses=losses,
                            sgd=optimizer
                        )
                    elif self.training_style == TrainingStyle.GOLD:
                        self.nlp.update(
                            docs,
                            goldparses,
                            drop=0.5,
                            losses=losses,
                            sgd=optimizer
                        )
                    else:
                        assert False
                self.logger.debug(f"Losses: {losses}")
            self.logger.info(f"Losses at final iteration: {losses}")

    ############################################ EVALUATE
    def evaluate(self, pipes_to_disable=['tagger', 'parser']):
        with self.nlp.disable_pipes(*pipes_to_disable):
            scorer = self.nlp.evaluate(
                [ ( d.sentence,
                    d.goldparse
                  ) for d in self.evaluation_data ],
                verbose=False
            )
            self.scorer = scorer
            self.scores = scorer.scores
            self.logger.info(f"Scores: {self.scores}")

    class ManualResults:
        def __init__(self):
            self.match = {}
            self.match['perfect'] = []
            self.match['whitespace'] = []
            self.match['character'] = []
            self.wrong = []
            self.extra = []
            self.missing = []
        def matches(self):
            # return [ m for type in self.match for m in self.match[type] ]
            return self.match['perfect'] + self.match['whitespace']
        def wrongs(self):
            # return self.wrong + self.extra
            return self.wrong + self.extra + self.match['character']
        def p(self):
            # How correct are the predictions?
            return len(self.matches()) / ( len(self.matches()) + len(self.wrongs()) ) * 100
        def r(self):
            # How complete are the predictions?
            return len(self.matches()) / ( len(self.matches()) + len(self.missing) ) * 100
        def __repr__(self):
            output = ""
            output += f"Correct tag: {len(self.match['perfect']+self.match['whitespace']+self.match['character'])} \n"
            output += f"Correct tag (offsets may differ in ws): {len(self.match['perfect']+self.match['whitespace'])} \n"
            output += f"Correct tag but wrong offsets: {len(self.match['character'])} \n"
            output += f"Wrong tag: {len(self.wrong)} \n"
            output += f"Extras: {len(self.extra)} \n"
            output += f"Missing: {len(self.missing)} \n"
            output += f"p={self.p()} \nr={self.r()}"
            return output

        class ComparisonPair:
            def __init__(self, predicted, correct):
                self.predicted = predicted
                self.correct = correct

    def evaluate_manually(self):
        results = NERer.ManualResults()
        for d in self.evaluation_data:
            if d.entities != []:
                self.__compare_results(
                    self.nlp(d.sentence),
                    d.entities,
                    results
                )
        self.scores_manual = results
        self.logger.info(f"Scores (manual): {self.scores_manual}")

    def __compare_results(self, doc, correct_ents, results):
        # this is the main function doing the comparison
        # the variable `results` will be updated with increased counters

        # these are the two lists
        # we map all manually annotated and predicted entities to the new data structure
        assert type(correct_ents[0]) is NamedEntity
        predicted_ents = [ NamedEntity.from_span(e, doc) for e in list(doc.ents) ]

        # sort both `correct_ents` and `predicted_ents` by their start position
        correct_ents.sort(key = lambda x: x.start)
        predicted_ents.sort(key = lambda x: x.start)

        # these are the two counters for the two lists
        i_corr = 0
        i_pred = 0

        while i_corr in range(len(correct_ents)) or i_pred in range(len(predicted_ents)):

            if i_corr == len(correct_ents): # we are out of manually annotated entities
                results.extra.append(NERer.ManualResults.ComparisonPair(predicted_ents[i_pred], None))
                i_pred += 1
                continue
            assert i_corr < len(correct_ents)

            if i_pred == len(predicted_ents): # we are out of predicted entities
                results.missing.append(NERer.ManualResults.ComparisonPair(None, correct_ents[i_corr]))
                i_corr += 1
                continue
            assert i_pred < len(predicted_ents)

            # compare the next element of `correct_ents` against the next element of `predicted_ents`
            next_correct = correct_ents[i_corr]
            next_prediction = predicted_ents[i_pred]

            if next_correct.is_before(next_prediction):
                results.missing.append(NERer.ManualResults.ComparisonPair(None, next_correct))
                i_corr += 1

            elif next_correct.is_after(next_prediction):
                results.extra.append(NERer.ManualResults.ComparisonPair(next_prediction, None))
                i_pred += 1

            elif next_correct.overlaps_with(next_prediction):
                if next_correct.label != next_prediction.label:
                    results.wrong.append(NERer.ManualResults.ComparisonPair(next_prediction, next_correct))

                elif next_correct.start == next_prediction.start and next_correct.end == next_prediction.end:
                    results.match['perfect'].append(NERer.ManualResults.ComparisonPair(next_prediction, next_correct))

                elif next_correct.overlaps_with_in_whitespace(next_prediction, doc.text):
                    results.match['whitespace'].append(NERer.ManualResults.ComparisonPair(next_prediction, next_correct))

                else:
                    results.match['character'].append(NERer.ManualResults.ComparisonPair(next_prediction, next_correct))

                i_corr += 1
                i_pred += 1
            else:
                assert False

    ############################################ SAVE
    def save(self, base_path):
        nowtime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = f"{base_path}/ner_apis_{nowtime}"
        self.logger.debug(f"Saving NERer as {save_path}.")
        self.saved_as = save_path
        self.__save_data(save_path)
        self.__save_nlp(save_path)
        self.__save_metadata(save_path)

    def __save_data(self, path):
        dir = Path(f'{path}/corpus')
        dir.mkdir(parents=True, exist_ok=True)
        # full data
        # this is hardly necessary as this data set is merely the union of training data and evaluation data
        # os.system(f"cp {self.corpus.path_to_data} {path}/corpus/")
        # vectors
        if self.path_to_vectors:
            path_vectors = f'{path}/corpus/vectors'
            p = subprocess.Popen(
                [
                    "cp", "-r", self.path_to_vectors, path_vectors
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE
            )
            p.wait()
            err = p.stderr.read().decode()
            if err:
                self.logger.error(err)
            out = p.stdout.read().decode()
            if out:
                self.logger.info(out)
            assert p.returncode == 0
            self.logger.debug(f"Saved vectors as {path_vectors}")
        # training data
        if len(self.training_data) > 0:
            path_trainset = f'{path}/corpus/trainset.pickle'
            pickle.dump(
                [ (x,y,z) for x, y, z, _ in self.training_data ],
                open(path_trainset, "wb")
            )
            self.logger.debug(f"Saved training set as {path_trainset}.")
        # evaluation data
        if len(self.evaluation_data) > 0:
            path_evalset = f'{path}/corpus/evalset.pickle'
            pickle.dump(
                [ (x,y,z) for x, y, z, _ in self.evaluation_data ],
                open(path_evalset, "wb")
            )
            self.logger.debug(f"Saved evaluation set as {path_evalset}.")
        # tokenizer abbreviations
        if self.abbreviations_file_path:
            path_abbr = f'{path}/corpus/abrevs_MS_OBL_manually_filtered.txt'
            p = subprocess.Popen(
                [
                    'cp', self.abbreviations_file_path, path_abbr
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE
            )
            p.wait()
            err = p.stderr.read().decode()
            if err:
                self.logger.error(err)
            out = p.stdout.read().decode()
            if out:
                self.logger.info(out)
            assert p.returncode == 0
            self.logger.debug(f"Saved abbreviations file as {path_abbr}.")

    def __save_nlp(self, path):
        path_model = f'{path}/nlp'
        nlp_dir = Path(path_model)
        nlp_dir.mkdir(parents=True, exist_ok=True)
        if self.nlp:
            self.nlp.to_disk(nlp_dir)
            self.logger.debug(f"Saved spacy model as {path_model}.")

    def __save_metadata(self, path):
        d = {}
        for key in self.map:
            if self.map[key] in self.__dict__:
                d[key] = self.__dict__[self.map[key]]
        path_dict = f'{path}/model_dict.pickle'
        pickle.dump(
            d,
            open(path_dict, 'wb')
        )
        self.logger.debug(f"Saved metadata as {path_dict}.")

class ModelType(Enum):
    NEWS_NEWS = 1 # load a news model and a default news ner
    NEWS_BLANK = 2 # load a news model and an empty ner
    BLANK_BLANK = 3 # load an empty model and an empty ner
    NEWS = 4 # add the news ner to the existing model
    BLANK = 5 # add an empty ner to the existing model

    def create_ner_pipe(self, nlp=None):
        # These are for the old models
        if self == ModelType.NEWS_NEWS:
            """Load a news model that contains the tagger and parser for sentencizing
            and the ner pipe."""
            nlp = spacy.load("de_core_news_md")
        elif self == ModelType.NEWS_BLANK:
            """Load a news model that contains the tagger and parser for sentencizing,
            but replace the ner pipe with a default ner pipe
            (actually I'm not sure what the default ner pipe is for the news model)."""
            nlp = spacy.load("de_core_news_md")
            ner = nlp.create_pipe("ner")
            nlp.replace_pipe(ner)
        elif self == ModelType.BLANK_BLANK:
            """Load spacy's default model for German."""
            nlp = spacy.blank("de")
            nlp.add_pipe(nlp.create_pipe("tagger"))
            nlp.add_pipe(nlp.create_pipe("parser"))
            nlp.add_pipe(nlp.create_pipe("ner"))
        # These are for the new models that always include the sentencizer
        elif self == ModelType.NEWS:
            """Add the ner pipe from spacy's news model to the model we already have."""
            assert nlp is not None
            # Add ner pipe from news model
            news = spacy.load('de_core_news_md')
            assert news.has_pipe('ner')
            nlp.add_pipe(news.get_pipe('ner'))
        elif self == ModelType.BLANK:
            """Add the default pipe that spacy creates for the model we already have
            to the model we already have."""
            assert nlp is not None
            # Add blank ner pipe
            ner = nlp.create_pipe("ner")
            if nlp.has_pipe('ner'):
                nlp.replace_pipe('ner', ner)
            else:
                nlp.add_pipe(ner)
        else:
            assert False
        assert nlp.has_pipe("ner")
        return nlp

NERer.ModelType = ModelType

class TrainingStyle(Enum):
    SIMPLE = 1
    GOLD = 2

NERer.TrainingStyle = TrainingStyle

def import_class(cl):
    (modulename, classname) = cl.rsplit('.', 1)
    m = __import__(modulename, globals(), locals(), [classname])
    return getattr(m, classname)
