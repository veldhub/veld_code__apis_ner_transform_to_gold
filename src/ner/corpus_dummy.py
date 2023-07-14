from .data import Datapoint, Dataset, NamedEntity

import logging

class PersonPlace:
    pass
class PersonInstitution:
    pass
class PersonPerson:
    pass

class Corpus:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._database = 'default'
        self.labels = []

    def extract_data(self, sentence_splitter_model, only_sentences_with_annotations=True):
        pass

    def _get_annotation_types(self, users, projects, text):
        pass

    def _collect_annotations(self, text, *when_conditions, filter_condition):
        pass

    def collect_annotations(self, text):
        """This calls this object's own `_collect_annotations` with no additional filter or when conditions. """
        return self._collect_annotations(text)

# ################################ corpora with LOC and ORG tags
class Corpus_LOC_ORG(Corpus):

    def __init__(self):
        super().__init__()
        self.labels = ["LOC", "ORG"]

    def _get_annotation_types(self, users, projects, text):
        pass

    def _collect_annotations(self, text, *when_conditions, filter_condition):
        """This calls `Corpus._collect_annotations`. """
        pass

class Corpus1(Corpus_LOC_ORG):
    def __init__(self):
        super().__init__() # self.labels
        self.description = """
            We define a trustworthy text in the following way:
            1. the text is linked to a vocabulary-base-class entry whose `name` contains "Haupttext", and
            2. the text contains at least one annotation such that
              1. the annotator (`user_added`) of this annotation is either `MKaiser` (`auth_user.id` 14) or `ABernad` (16), and
              2. the annotation project (`annotation_project_id`) that this annotation belongs to is in the set [3, 4, 5, 10, 11, 16], and
              3. one of the annotated entities is `apis_relations.personplace` (`django_content_type.id` 40) or `apis_relations.personinstitution` (41).
        """
        self.annotation_projects = [3, 4, 5, 10, 11, 16]
        self.annotators = [14, 16]

    def collect_texts(self):
        pass

class Corpus2(Corpus_LOC_ORG):

    def __init__(self):
        super().__init__() # self.labels as set in Corpus_LOC_ORG.__init__()
        self.description = ""
        self.annotation_projects = [3]
        self.annotators = [14, 16]
        self.text_length_cutoff = 300
        self._database = 'apis_apis_edit' # TODO: this is not desirable, but it may be necessary to do UNION with corpus 3, which must use the database 'apis_apis_edit'

    def _collect_texts(self):
        pass

    def collect_texts(self):
        pass

class Corpus3(Corpus_LOC_ORG):

    def __init__(self):
        super().__init__() # self.labels
        self.description = ""
        self.annotation_projects = [16]
        self.annotators = [14,16]
        self.collection = [22]
        self.text_length_cutoff = 300
        self._database = 'apis_apis_edit'

    def _collect_texts(self):
        pass

    def collect_texts(self):
        pass

class Corpus4(Corpus_LOC_ORG):
    def __init__(self):
        super().__init__()
        self.description = """
            Union of corpus 2 and corpus 3.
        """
        self.annotators = list(set(Corpus2().annotators + Corpus3().annotators))
        self.annotation_projects = list(set(Corpus2().annotation_projects + Corpus3().annotation_projects))

    def collect_texts(self):
        pass

# ################################ corpora with PER, LOC, and ORG tags
class Corpus_PER_LOC_ORG(Corpus_LOC_ORG):

    def __init__(self):
        super().__init__() # self.labels
        self.labels.append("PER")

    def _get_annotation_types(self, users, projects, text):
        pass

    def _collect_annotations(self, text, *when_conditions, filter_condition):
        pass

class Corpus2Per(Corpus_PER_LOC_ORG, Corpus2):
    """Inherits
        `Corpus_PER_LOC_ORG._get_annotation_types`,
        `Corpus.collect_annotations`, which calls `Corpus_PER_LOC_ORG._collect_annotations`,
        `Corpus2.collect_texts`, which calls `Corpus2Per._collect_texts`
    """
    def __init__(self):
        self.labels = []
        super().__init__() # self.labels as set in Corpus_PER_LOC_ORG.__init__()
        super(Corpus2, self).__init__() # self.annotators etc. as in Corpus2.__init__()
        self.description = ""
        self._database = 'apis_apis_edit' # TODO: maybe this should be removed. this corpus is actually meant to execute on 'apis_apis'

    def _collect_texts(self):
        pass

class Corpus3Per(Corpus_PER_LOC_ORG, Corpus3):
    def __init__(self):
        self.labels = []
        super().__init__() # self.labels as set in Corpus_PER_LOC_ORG.__init__()
        super(Corpus3, self).__init__() # self.annotators etc. as in Corpus3.__init__()
        self._database = 'apis_apis_edit'

    def _collect_texts(self):
        pass

class Corpus4Per(Corpus_PER_LOC_ORG, Corpus4):
    def __init__(self):
        super().__init__()
            # This will call, in this order:
            # Corpus_PER_LOC_ORG.__init__() -- set PER label
            # Corpus4.__init__() -- set annotators etc.
            # Corpus_LOC_ORG.__init__() -- set LOC and ORG labels
            # Corpus.__init__() -- set logger

    def collect_texts(self):
        pass

class Corpus_Prodigy(Corpus):
    def __init__(self):
        super().__init__()

class Corpus5(Corpus_Prodigy, Corpus_LOC_ORG):
    def __init__(self):
        super().__init__()

Corpus5Per = Corpus5

class Corpus6(Corpus_LOC_ORG):
    def __init__(self):
        super().__init__()

class Corpus6Per(Corpus_PER_LOC_ORG):
    def __init__(self):
        super().__init__()

# ################################ todo: corpora with relation tags
class Corpus_Relations(Corpus_PER_LOC_ORG):
    def __init__(self):
        super().__init__()
        self.labels = []

class Corpus4Rel(Corpus_Relations, Corpus4):
    def __init__(self):
        super().__init__()
        self.labels = [
            'ORG-5660',
            'PER-5430',
            'ORG-5658',
            'ORG-5777',
            'PER-5781',
            'ORG-5656',
            'ORG-5675',
            'LOC-5787',
            'ORG-5691',
            'ORG-5642',
            'PER-5414',
            'PER-5801',
            'LOC-5401',
            'PER-5769',
            'ORG-5648',
            'ORG-5617',
            'ORG-5812',
            'PER-5416',
            'PER-5417',
            'ORG-5652',
            'LOC-5392',
            'ORG-5395',
            'ORG-5760',
            'LOC-5390',
            'ORG-5645',
            'ORG-5643',
            'PER-5423',
            'ORG-5674',
            'PER-5415',
            'LOC-5402',
            'PER-5421',
            'ORG-5622',
            'ORG-5630',
            'ORG-5612',
            'ORG-5778',
            'ORG-5610',
            'ORG-5611',
            'ORG-5616',
            'PER-5418',
            'PER-5800',
            'PER-5775',
            'ORG-5620',
            'ORG-5646',
            'PER-5425',
            'ORG-5791',
            'ORG-5621',
            'ORG-5689',
            'PER-5424',
            'ORG-5396',
            'ORG-5686',
            'PER-5432',
            'ORG-5679',
            'ORG-5806',
            'ORG-5683',
            'LOC-5388',
            'ORG-5640',
            'ORG-5613',
            'ORG-5776',
            'ORG-5792',
            'ORG-5676',
            'LOC-5399',
            'LOC-5403',
            'ORG-5677',
            'PER-5759',
            'ORG-5879',
            'ORG-5697',
            'ORG-5624',
            'LOC-5391',
            'ORG-5667',
            'PER-5785',
            'ORG-5688',
            'PER-5412',
            'ORG-5654',
            'ORG-5662',
            'ORG-5690',
            'ORG-5701',
            'ORG-5631',
            'PER-5715',
            'ORG-5398',
            'LOC-5400',
            'PER-5411',
            'ORG-5634',
            'ORG-5698',
            'ORG-5657',
            'ORG-5672',
            'PER-5413',
            'PER-5422',
            'PER-5420',
            'PER-5783',
            'ORG-5651',
            'ORG-5682',
            'PER-5428',
            'ORG-5659'
        ]

class Corpus_CoarseRelations(Corpus_Relations):
    def __init__(self):
        super().__init__()

class Corpus4CRel(Corpus_CoarseRelations, Corpus4Per):
    def __init__(self):
        super().__init__()
        self.labels = [
            'ORG-5398',
            'PER-5769',
            'PER-5775',
            'PER-5413',
            'ORG-5657',
            'LOC-5388',
            'ORG-5639',
            'LOC-5399',
            'PER-5416',
            'PER-5420',
            'PER-5781',
            'ORG-5395',
            'PER-5412',
            'LOC-5391',
            'ORG-5611',
            'PER-5421',
            'ORG-5622',
            'ORG-5760',
            'ORG-5679',
            'PER-5759',
            'LOC-5787',
            'ORG-5701',
            'PER-5430',
            'ORG-5618',
            'PER-5801'
        ]
