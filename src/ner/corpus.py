from .data import Datapoint, Dataset, NamedEntity

from apis_core.apis_relations.models import PersonPerson, PersonPlace, PersonInstitution
from apis_core.apis_metainfo.models import TempEntityClass, Text
from apis_core.apis_vocabularies.models import VocabsBaseClass
from apis_highlighter.models import Annotation

from django.db.models import BooleanField, CharField, IntegerField
from django.db.models import Exists, Subquery
from django.db.models import Case, ExpressionWrapper, F, OuterRef, Q, Value, When
from django.db.models.functions import Concat, Length
import jsonlines
import logging
import random
import spacy
import string

class Corpus:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._database = 'default'
        self.labels = []

    def extract_data(self, sentence_splitter_model, only_sentences_with_annotations=True):
        """Returns a Dataset containing sentences and their annotations."""
        self.logger.debug(f"Extracting data from corpus {self.__class__.__name__} ...")
        data = Dataset()
        num_annotations = 0
        texts = self.collect_texts()
        num_texts_total = len(texts)
        text_counter = 0
        for text in texts:
            text_counter += 1
            self.logger.debug(f"Working on text #{text_counter}/{num_texts_total}; id={text.pk} ...")
            annotations = self.collect_annotations(text)
            self.logger.debug(f"Found {len(annotations)} annotations in this text.")
            if len(annotations) == 0 and only_sentences_with_annotations:
                continue
            doc = sentence_splitter_model.sentencize(text.text)
            if doc.is_sentenced:
                sents = list(doc.sents)
                # self.logger.debug(f"Text id {text.id} contains {len(sents)} sentences in total.")
            else:
                sents = [ doc[0:len(doc)] ]
            num_sents_total = len(sents)
            num_sents_with_annotations = 0
            sentence_counter = 0
            for sent in sents:
                self.logger.debug(f"Working on sentence #{sentence_counter}/{num_sents_total} ...")
                anns_in_current_sent = annotations.filter(
                    start__gte = sent.start_char,
                    end__lt = sent.end_char
                )
                entities = merge_annotations(sent, anns_in_current_sent)
                num_annotations += len(entities)
                if len(entities) > 0 or not only_sentences_with_annotations:
                    data.append(
                        Datapoint(
                            sent.text,
                            entities,
                            doc=sentence_splitter_model.sentencize(sent.text)
                            # This should always yield a doc with a single sentence,
                            # but this is not the case! Sometimes we get a doc with
                            # 2 sentences! (This is not a big problem because calling
                            # ents_to_doc and training with this doc will work no matter
                            # how many sentences are in this doc. But it shows that the
                            # sentence splitting has substantial problems.)
                        )
                    )
                    self.logger.debug(f"Added this sentence with {len(entities)} entities.")
                    if len(entities) > 0:
                        num_sents_with_annotations += 1
                sentence_counter += 1
            if not only_sentences_with_annotations:
                self.logger.debug(f"Out of these {len(sents)} sentences, {num_sents_with_annotations} had annotations.")
        self.logger.info(f"Extracted a dataset with {len(data)} sentences, containing a total of {num_annotations} annotations.")
        return data

    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        PerPer = PersonPerson.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))
        PerPl = PersonPlace.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))
        PerIn = PersonInstitution.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))

        anns = Annotation.objects.using(self._database).filter(
            text = text,
            user_added__in = self.annotators,
            annotation_project__in = self.annotation_projects,
        )
        anns = anns.annotate(
            isPersonPerson = Exists(PerPer),
            isPersonPlace = Exists(PerPl),
            isPersonInstitution = Exists(PerIn),
        )
        for column_name, expression in annotate:
            anns = anns.annotate(**{column_name: expression})
        anns = anns.filter(filter)
        anns = anns.order_by('start', 'end')
        return anns

    def collect_annotations(self, text):
        """This calls this object's own `_collect_annotations` with no additional filter or when conditions. """
        self.logger.debug(f"Collecting annotations for text {text.pk} from corpus {self.__class__.__name__} ...")
        return self._collect_annotations(text)

def merge_annotations(sentence, annotations):
    """
    As we iterate over the annotations we found in the db, we will need to
    merge overlapping annotations (spacy doesn't allow this).
    We merge along the following criteria:
    1. If annotations disagree in their tag, prefer 'ORG' over 'LOC'.
    2. Else, use the longest offsets. Then strip.
    """
    entities = [] # this is the list of Entities this function returns
    overlapping_anns = OverlappingAnnotations([], sentence=sentence)
    for ann in annotations:
        # for the first ann, don't do any comparisons
        if len(overlapping_anns) == 0:
            overlapping_anns.append(ann)
            continue
        # for any later ann:
        # check whether this ann overlaps with the tentative ann
        # case 1: this annotation doesn't overlap with the previous one :)
        if overlapping_anns.end() < ann.start:
            # so we keep the previous tentative one as final choice.
            entities.append(overlapping_anns.create_entity())
            overlapping_anns.reset()
            overlapping_anns.append(ann)
        # case 2: this annotation overlaps with the previous one!
        else:
            # so we store it in the current set of overlapping entities
            overlapping_anns.append(ann)
    # in the last run of the loop, we have a set of overlapping annotation that has not yet been saved
    # so we save it now
    if len(overlapping_anns) > 0:
        entities.append(overlapping_anns.create_entity())

    return entities

class OverlappingAnnotations(list):
    """This class manages overlapping annotations, which need to be 'merged'
    into a single annotation for spacy's NER."""
    def __init__(self, *args, sentence=None):
        if len(args) > 0:
            arg = args[0]
        else:
            arg = []
        super(OverlappingAnnotations, self).__init__(arg)
        if sentence:
            assert type(sentence) is spacy.tokens.span.Span
        self.sentence = sentence
        self.logger = logging.getLogger(__name__)

    def start(self):
        return min([ a.start for a in self ])

    def end(self):
        return max([ a.end for a in self ])

    def lrstrip(self):
        """Return the leftmost non-whitespace start index and the rightmost non-whitespace end index."""
        leftmost = self.start()
        rightmost = self.end()
        rightmost = min(rightmost, self.sentence.end_char) # because some annotations are longer than the sentence
        # make the indices relative to the sentence, not to the text
        leftmost -= self.sentence.start_char
        rightmost -= self.sentence.start_char
        # advance the indices to non-whitespace characters
        while self.sentence.text[leftmost] in '¿„'+string.whitespace+string.punctuation and leftmost < len(self.sentence.text):
            leftmost += 1
        while self.sentence.text[rightmost-1] in ',“’'+string.whitespace+string.punctuation and rightmost > 1:
            rightmost -= 1
        return leftmost, rightmost

    def tag(self):
        tags = set([ a.ner_tag for a in self ])
        if len(tags) == 1:
            return list(tags)[0]
        elif 'ORG' in tags:
            return 'ORG'
        else:
            # determine whether one tag is a subtype of the others
            # if yes, use the most specific one

            # if not, pick one at random
            return list(tags)[0]

    def create_entity(self):
        """Create one single NamedEntity that contains the correct offsets and label."""
        # get the left-most start index and the right-most end index that are not whitespace
        start, end = self.lrstrip()
        entity = NamedEntity(
            start,
            end,
            self.tag(),
            text = self.sentence.text[start:end],
            context = self.sentence.text,
            annotation_ids = [ a.pk for a in self ]
        )
        self.logger.debug(f"Created an entity with text '{entity.text}' of type '{entity.label}'.")
        return entity

    def reset(self):
        self.__init__([], sentence=self.sentence)

# ################################ corpora with LOC and ORG tags
class Corpus_LOC_ORG(Corpus):

    def __init__(self):
        super().__init__()
        self.labels = ["LOC", "ORG"]

    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        """This calls `Corpus._collect_annotations`. """
        return super()._collect_annotations(
            text,
            filter = Q(isPersonPlace=True) | Q(isPersonInstitution=True) | filter,
            annotate = [
                (
                    'content_type',
                    Case(
                        When(isPersonPerson=True, then=Value("PER")),
                        When(isPersonPlace=True, then=Value("LOC")),
                        When(isPersonInstitution=True, then=Value("ORG")),
                        output_field = CharField()
                    )
                ),
                (
                    'ner_tag',
                    F('content_type')
                )
            ] + annotate
        )

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
        texts = Text.objects.annotate(
            hasPersonPlaceAnnotations = Exists(
                self._get_annotation_types(self.annotators, self.annotation_projects, OuterRef('pk')).filter(isPersonPlace=True)),
            hasPersonInstitutionAnnotations = Exists(
                self._get_annotation_types(self.annotators, self.annotation_projects, OuterRef('pk')).filter(isPersonInstitution=True)),
        ).filter(
            kind__name__icontains = "Haupttext",
            hasPersonPlaceAnnotations = True,
            hasPersonInstitutionAnnotations = True,
        )
        return texts

class Corpus2(Corpus_LOC_ORG):

    def __init__(self):
        super().__init__() # self.labels as set in Corpus_LOC_ORG.__init__()
        self.description = ""
        self.annotation_projects = [3]
        self.annotators = [14, 16]
        self.text_length_cutoff = 300
        self._database = 'apis_apis_edit' # TODO: this is not desirable, but it may be necessary to do UNION with corpus 3, which must use the database 'apis_apis_edit'

    def _collect_texts(self):
        texts2 = Text.objects.using(self._database).annotate(
            hasPersonPersonAnnotations = Value(None, output_field=BooleanField()),
            hasPersonPlaceAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPlace=True))
            ),
            hasPersonInstitutionAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonInstitution=True))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Value(None, output_field=BooleanField()),
                # we need this because for corpus 4, we will call UNION with texts3,
                # and texts3 annotates this field because it needs to filter for Exists
                # and since we are using Django 2.1, filtering for an Exists can only be done via annotate
        ).filter(
            kind__name__icontains="Haupttext",
        )
        # short texts
        short = texts2.filter(
            Q(hasPersonPlaceAnnotations=True) | Q(hasPersonInstitutionAnnotations=True),
            text_charlength__lte = self.text_length_cutoff,
        )
        # long texts
        long = texts2.filter(
            Q(hasPersonPlaceAnnotations=True) & Q(hasPersonInstitutionAnnotations=True),
            text_charlength__gt = self.text_length_cutoff,
        )
        return short, long

    def collect_texts(self):
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        short, long = self._collect_texts()
        return short.union(long)

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
        texts3 = Text.objects.using(self._database).annotate(
            hasPersonPersonAnnotations = Value(None, output_field=BooleanField()),
            hasPersonPlaceAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPlace=True))
            ),
            hasPersonInstitutionAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonInstitution=True))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Exists(TempEntityClass.objects.using(self._database).filter(
                text__pk = OuterRef('pk'),
                collection__pk__in = self.collection
            ))
        ).filter(
            kind__name__icontains="Haupttext",
            belongsToCollection22 = True
        )
        # short texts
        short = texts3.filter(
            Q(hasPersonPlaceAnnotations=True) | Q(hasPersonInstitutionAnnotations=True),
            text_charlength__lte = self.text_length_cutoff,
        )
        # long texts
        long = texts3.filter(
            Q(hasPersonPlaceAnnotations=True) & Q(hasPersonInstitutionAnnotations=True),
            text_charlength__gt = self.text_length_cutoff,
        )
        return short, long

    def collect_texts(self):
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        short, long = self._collect_texts()
        return short.union(long)

class Corpus4(Corpus_LOC_ORG):
    def __init__(self):
        super().__init__()
        self.description = """
            Union of corpus 2 and corpus 3.
        """
        self.annotators = list(set(Corpus2().annotators + Corpus3().annotators))
        self.annotation_projects = list(set(Corpus2().annotation_projects + Corpus3().annotation_projects))

    def collect_texts(self):
        """For some reason, using `short2.union(long2, ...)` doesn't give us the right results."""
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        c2 = Corpus2()
        c3 = Corpus3()
        assert c2._database == c3._database
        short2, long2 = c2._collect_texts()
        short3, long3 = c3._collect_texts()
        bigunion = short2.values('pk').union(long2.values('pk'), short3.values('pk'), long3.values('pk'))
        return Text.objects.using(c2._database).filter(pk__in=[ x['pk'] for x in bigunion ])

# ################################ corpora with PER, LOC, and ORG tags
class Corpus_PER_LOC_ORG(Corpus_LOC_ORG):

    def __init__(self):
        super().__init__() # self.labels
        self.labels.append("PER")

    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        """This calls `Corpus_LOC_ORG._collect_annotations`. """
        return super()._collect_annotations(
            text,
            filter = Q(isPersonPerson=True) | filter,
            annotate = annotate
        )

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
        texts2 = Text.objects.using(self._database).annotate(
            hasPersonPersonAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPerson=True))
            ),
            hasPersonPlaceAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPlace=True))
            ),
            hasPersonInstitutionAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonInstitution=True))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Value(None, output_field=BooleanField()),
                # we need this because for corpus 4, we will call UNION with texts3,
                # and texts3 annotates this field because it needs to filter for Exists
                # and since we are using Django 2.1, filtering for an Exists can only be done via annotate
        ).filter(
            kind__name__icontains="Haupttext",
        )
        # short texts
        short = texts2.filter(
            Q(hasPersonPersonAnnotations=True) | Q(hasPersonPlaceAnnotations=True) | Q(hasPersonInstitutionAnnotations=True),
            text_charlength__lte = self.text_length_cutoff,
        )
        # long texts
        long = texts2.filter(
            Q(hasPersonPersonAnnotations=True) | (
                Q(hasPersonPlaceAnnotations=True) & Q(hasPersonInstitutionAnnotations=True)
            ),
            text_charlength__gt = self.text_length_cutoff,
        )
        return short, long

class Corpus3Per(Corpus_PER_LOC_ORG, Corpus3):
    def __init__(self):
        self.labels = []
        super().__init__() # self.labels as set in Corpus_PER_LOC_ORG.__init__()
        super(Corpus3, self).__init__() # self.annotators etc. as in Corpus3.__init__()
        self._database = 'apis_apis_edit'

    def _collect_texts(self):

        texts3 = Text.objects.using(self._database).annotate(
            hasPersonPersonAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPerson=True))
            ),
            hasPersonPlaceAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonPlace=True))
            ),
            hasPersonInstitutionAnnotations = Exists(
                self._collect_annotations(OuterRef('pk'), filter=Q(isPersonInstitution=True))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Exists(TempEntityClass.objects.using(self._database).filter(
                text__pk = OuterRef('pk'),
                collection__pk__in = self.collection
            ))
        ).filter(
            kind__name__icontains="Haupttext",
            belongsToCollection22 = True
        )
        # short texts
        short = texts3.filter(
            Q(hasPersonPersonAnnotations=True) | Q(hasPersonPlaceAnnotations=True) | Q(hasPersonInstitutionAnnotations=True),
            text_charlength__lte = self.text_length_cutoff,
        )
        # long texts
        long = texts3.filter(
            Q(hasPersonPersonAnnotations=True) | (Q(hasPersonPlaceAnnotations=True) & Q(hasPersonInstitutionAnnotations=True)),
            text_charlength__gt = self.text_length_cutoff,
        )
        return short, long

class Corpus4Per(Corpus_PER_LOC_ORG, Corpus4):
    def __init__(self):
        super().__init__()
            # This will call, in this order:
            # Corpus_PER_LOC_ORG.__init__() -- set PER label
            # Corpus4.__init__() -- set annotators etc.
            # Corpus_LOC_ORG.__init__() -- set LOC and ORG labels
            # Corpus.__init__() -- set logger

    def collect_texts(self):
        """For some reason, using `short2.union(long2, ...)` doesn't give us the right results."""
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        c2p = Corpus2Per()
        c3p = Corpus3Per()
        assert c2p._database == c3p._database
        short2, long2 = c2p._collect_texts()
        short3, long3 = c3p._collect_texts()
        bigunion = short2.values('pk').union(long2.values('pk'), short3.values('pk'), long3.values('pk'))
        return Text.objects.using(c2p._database).filter(pk__in=[ x['pk'] for x in bigunion ])

# ################################

class Corpus_PER(Corpus):
    def __init__(self):
        super().__init__()
        self.labels = ["PER"]
    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        """This calls `Corpus._collect_annotations`. """
        return super()._collect_annotations(
            text,
            filter = Q(isPersonPerson=True) | filter,
            annotate = [
                (
                    'content_type',
                    Case(
                        When(isPersonPerson=True, then=Value("PER")),
                        output_field = CharField()
                    )
                ),
                (
                    'ner_tag',
                    F('content_type')
                )
            ] + annotate
        )

class Corpus2PerOnly(Corpus_PER):

    def __init__(self):
        super().__init__() # self.labels as set in Corpus_PER.__init__()
        self.description = ""
        self.annotation_projects = [3]
        self.annotators = [14, 16]
        self.text_length_cutoff = 300
        self._database = 'apis_apis_edit' # TODO: this is not desirable, but it may be necessary to do UNION with corpus 3, which must use the database 'apis_apis_edit'

    def _collect_texts(self):
        texts2 = Text.objects.using(self._database).annotate(
            countPersonPersonAnnotations = Count(
                self._collect_annotations(OuterRef('pk'))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Value(None, output_field=BooleanField()),
                # we need this because for corpus 4, we will call UNION with texts3,
                # and texts3 annotates this field because it needs to filter for Exists
                # and since we are using Django 2.1, filtering for an Exists can only be done via annotate
        ).filter(
            kind__name__icontains="Haupttext",
        )
        # short texts
        short = texts2.filter(
            text_charlength__lte = self.text_length_cutoff,
            countPersonPersonAnnotations__gte = 1,
        )
        # long texts
        long = texts2.filter(
            text_charlength__gt = self.text_length_cutoff,
            countPersonPersonAnnotations__gte = 2,
        )
        return short, long

class Corpus3PerOnly(Corpus_PER):
    def __init__(self):
        super().__init__() # self.labels
        self.description = ""
        self.annotation_projects = [16]
        self.annotators = [14,16]
        self.collection = [22]
        self.text_length_cutoff = 300
        self._database = 'apis_apis_edit'

    def _collect_texts(self):
        texts3 = Text.objects.using(self._database).annotate(
            countPersonPersonAnnotations = Count(
                self._collect_annotations(OuterRef('pk'))
            ),
            text_charlength = Length(F('text')),
            belongsToCollection22 = Exists(TempEntityClass.objects.using(self._database).filter(
                text__pk = OuterRef('pk'),
                collection__pk__in = self.collection
            ))
        ).filter(
            kind__name__icontains="Haupttext",
            belongsToCollection22 = True
        )
        # short texts
        short = texts3.filter(
            text_charlength__lte = self.text_length_cutoff,
            countPersonPersonAnnotations__gte = 1,
        )
        # long texts
        long = texts3.filter(
            text_charlength__gt = self.text_length_cutoff,
            countPersonPersonAnnotations__gte = 2,
        )
        return short, long

class Corpus4PerOnly(Corpus_PER):
    def __init__(self):
        super().__init__()
        self.description = """
            Union of corpus 2 and corpus 3.
        """
        self.c2 = Corpus2PerOnly()
        self.c3 = Corpus3PerOnly()
        self.annotators = list(set(self.c2.annotators + self.c3.annotators))
        self.annotation_projects = list(set(self.c2.annotation_projects + self.c3.annotation_projects))

    def collect_texts(self):
        """For some reason, using `short2.union(long2, ...)` doesn't give us the right results."""
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        assert self.c2._database == self.c3._database
        short2, long2 = self.c2._collect_texts()
        short3, long3 = self.c3._collect_texts()
        bigunion = short2.values('pk').union(long2.values('pk'), short3.values('pk'), long3.values('pk'))
        return Text.objects.using(c2._database).filter(pk__in=[ x['pk'] for x in bigunion ])

# ################################
class TempAnnotation(Annotation):
    ner_tag = CharField(max_length=255, blank=True, null=True)
    class Meta:
        app_label = 'apis_highlighter'

class TempAnnotationObjects(list):
    """This is a list that is extended by a function `filter` that emulates `QuerySet.filter` over Annotations. """
    def __init__(self, *args):
        if len(args) > 0:
            arg = args[0]
        else:
            arg = []
        super(TempAnnotationObjects, self).__init__(arg)

    def filter(self, text=None, start__gte=None, end__lt=None):
        result = self
        if text:
            result = filter(lambda ta: ta.text == text, result)
        if start__gte:
            result = filter(lambda ta: ta.start >= start__gte, result)
        if end__lt:
            result = filter(lambda ta: ta.end < end__lt, result)
        return TempAnnotationObjects(result)

class Corpus_Prodigy(Corpus):
    """Reads the corpus from a list of JSONL files instead of accessing the database."""

    def __init__(self):
        super().__init__()
        if not self.jsonl_paths:
            self.jsonl_paths = []
        self.dataset = Dataset()
        self.annotations = TempAnnotationObjects()
        self.__jsonl_is_parsed = False

    def __parse_jsonl(self):
        self.logger.debug(f"Parsing JSONL files for corpus {self.__class__.__name__} ...")
        # parse the jsonl files and extract a Dataset filled with Datapoints.
        # also extract the annotations into TempAnnotation objects,
        # since the annotations are not persisted in the database and we will need to filter them by text_id.
        for path in self.jsonl_paths:
            with jsonlines.open(path) as reader:
                for obj in reader:
                    # get the text from the jsonl
                    sentence = obj['text']
                    # get the person id from the jsonl
                    person_pk = obj['meta']['person pk']
                    # look for the text in the database
                    text_objects = Text.objects.filter(text__contains=sentence)
                    text_objects = text_objects.annotate(personmatches=Exists(TempEntityClass.objects.filter(text__pk=OuterRef('pk'), pk=person_pk)))
                    text_objects = text_objects.filter(personmatches=True)
                    text_object = text_objects[0]
                    # read the entities from the jsonl
                    ents = []
                    for s in obj['spans']:
                        # create a named entity to go into the dataset
                        ne = NamedEntity(
                            s['start'], s['end'], s['label'],
                            text=sentence[s['start']:s['end']],
                            context=sentence
                        )
                        ents.append(ne)
                        # create a temp annotation so that we can mock being a database corpus rather than a jsonl corpus
                        ta = TempAnnotation(
                            start=s['start'], end=s['end'], ner_tag=s['label'],
                            text=text_object
                        )
                        self.annotations.append(ta)
                    self.dataset.append(Datapoint(sentence, ents))
        self.__jsonl_is_parsed = True

    def collect_texts(self):
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        if not self.__jsonl_is_parsed: self.__parse_jsonl()
        bigQ = Q()
        for d in self.dataset:
            bigQ = bigQ | Q(text__contains = d.sentence)
        return Text.objects.filter(bigQ)

    def collect_annotations(self, text):
        self.logger.debug(f"Collecting annotations for text {text.pk} from corpus {self.__class__.__name__} ...")
        if not self.__jsonl_is_parsed: self.__parse_jsonl()
        result = self.annotations.filter(text=text)
        result = sorted(result, key=lambda ta: (ta.start, ta.end))
        return TempAnnotationObjects(result)

class Corpus5(Corpus_Prodigy, Corpus_LOC_ORG):
    """This is the corpus that was annotated in February 2020 by the interns."""
    def __init__(self):
        self.jsonl_paths = [
            '../spacy-ner/prodigy_data/apis_1_annotations.jsonl',
            '../spacy-ner/prodigy_data/apis_2_annotations.jsonl',
            '../spacy-ner/prodigy_data/apis_3_annotations.jsonl'
        ]
        super().__init__()

Corpus5Per = Corpus5
# For backwards compatibility reasons only:
# The NER models trained on 2020-04-28 were told they are trained over "Corpus5Per".
# In fact, Corpus 5 (the corpus that was annotated during February 2020) does not contain the label PER."""

class Corpus6(Corpus_LOC_ORG):
    def __init__(self):
        super().__init__()
        self.c4 = None
        self.c5 = None

    def collect_texts(self):
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        if not self.c4: self.c4 = Corpus4()
        if not self.c5: self.c5 = Corpus5()
        c4_texts = self.c4.collect_texts() # returns a QuerySet
        c5_texts = self.c5.collect_texts() # returns a QuerySet
        return c4_texts.union(c5_texts)

    def collect_annotations(self, text):
        self.logger.debug(f"Collecting annotations for text {text.pk} from corpus {self.__class__.__name__} ...")
        if not self.c4: self.c4 = Corpus4()
        if not self.c5: self.c5 = Corpus5()
        c4_ann = self.c4.collect_annotations(text) # returns a QuerySet
        c5_ann = self.c5.collect_annotations(text) # returns a TempAnnotationObjects list
        return TempAnnotationObjects(list(c4_ann) + c5_ann)

class Corpus6Per(Corpus_PER_LOC_ORG):
    """This is the union of Corpus4Per and Corpus5."""
    def __init__(self):
        super().__init__()
        self.c4 = None
        self.c5 = None

    def collect_texts(self):
        self.logger.debug(f"Collecting texts from corpus {self.__class__.__name__} ...")
        if not self.c4: self.c4 = Corpus4Per()
        if not self.c5: self.c5 = Corpus5()
        c4_texts = self.c4.collect_texts() # returns a QuerySet
        c5_texts = self.c5.collect_texts() # returns a QuerySet
        return c4_texts.union(c5_texts)

    def collect_annotations(self, text):
        self.logger.debug(f"Collecting annotations for text {text.pk} from corpus {self.__class__.__name__} ...")
        if not self.c4: self.c4 = Corpus4Per()
        if not self.c5: self.c5 = Corpus5()
        c4_ann = self.c4.collect_annotations(text) # returns a QuerySet
        c5_ann = self.c5.collect_annotations(text) # returns a TempAnnotationObjects list
        return TempAnnotationObjects(list(c4_ann) + c5_ann)

# ################################ todo: corpora with relation tags
class Corpus_Relations(Corpus_PER_LOC_ORG):
    def __init__(self):
        super().__init__()

        labels = set()
        texts = self.collect_texts()
        for t in texts:
            anns = self.collect_annotations(t)
            labels.update(anns.values_list('ner_tag', flat=True))
        self.labels = list(labels)

    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        """This calls `Corpus_PER_LOC_ORG._collect_annotations`."""
        PerPer = PersonPerson.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))
        PerPl = PersonPlace.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))
        PerIn = PersonInstitution.objects.using(self._database).filter(annotation__pk=OuterRef('pk'))
        return super()._collect_annotations(
            text,
            filter = filter,
            annotate = [
                (
                    'relation_type_id',
                    Case(
                        When(isPersonPerson=True, then=Subquery(PerPer.values('relation_type__pk'))),
                        When(isPersonPlace=True, then=Subquery(PerPl.values('relation_type__pk'))),
                        When(isPersonInstitution=True, then=Subquery(PerIn.values('relation_type__pk'))),
                        output_field = IntegerField()
                    )
                ),
                (
                    'relation_type_name',
                    Case(
                        When(isPersonPerson=True, then=Subquery(PerPer.values('relation_type__name'))),
                        When(isPersonPlace=True, then=Subquery(PerPl.values('relation_type__name'))),
                        When(isPersonInstitution=True, then=Subquery(PerIn.values('relation_type__name'))),
                        output_field = CharField()
                    )
                ),
                (
                    'relation_type',
                    Concat(
                        F('content_type'),
                        Value('-'),
                        ExpressionWrapper(F('relation_type_id'), output_field=CharField()),
                        # Value('-'),
                        # F('relation_type_name')
                    )
                ),
                (   # this overwrites the 'ner_tag' annotation from Corpus_LOC_ORG
                    'ner_tag',
                    F('relation_type')
                ),
            ] + annotate
        )

class Corpus4Rel(Corpus_Relations, Corpus4Per):
    def __init__(self):
        super().__init__()

class Corpus_CoarseRelations(Corpus_Relations):
    def __init__(self):
        super().__init__()

    def _collect_annotations(self, text, filter=Q(), annotate=[]):
        """This calls `Corpus_Relations._collect_annotations`."""
        # First we create a lookup for relation names (stored in VocabsBaseClass),
        # where we get a direct mapping from the relation's ID to the ID of its topmost ancestor.
        # The definition of this is recursive, and so the proper way to do this would be
        # a recursive SQL statement. However, Django does not seem to support this except for raw SQL,
        # and I have not been able to find a way to use a raw SQL query as Subquery.
        # Having run out of options attempting to do this, I will now instead
        # make use of a property of this particular dataset, namely that no
        # hierarchical relation system goes deeper than 3 levels.
        if not self.vbc:
            self.vbc = VocabsBaseClass.objects.select_related('parent_class').annotate(
                p1 = Case(
                    When(
                        parent_class__isnull=True,
                        then=F('pk')
                    ),
                    default = Value(0),
                    output_field = IntegerField()
                ),
                p2 = Case(
                    When(
                        p1=0,
                        parent_class__parent_class__isnull=True,
                        then=F('parent_class_id')
                    ),
                    default = Value(0),
                    output_field = IntegerField()
                ),
                p3 = Case(
                    When(
                        p1=0,
                        p2=0,
                        parent_class__parent_class__parent_class__isnull=True,
                        then=Subquery(
                            VocabsBaseClass.objects.filter(pk=OuterRef('parent_class_id')).values('parent_class_id')
                        )
                    ),
                    default = Value(0),
                    output_field = IntegerField()
                ),
                top_relation_id = Case(
                    When(p1__gt=0, then=F('p1')),
                    When(p2__gt=0, then=F('p2')),
                    When(p3__gt=0, then=F('p3')),
                    default = Value(0),
                    output_field = IntegerField()
                )
            )
        return super()._collect_annotations(
            text,
            filter = filter,
            annotate = [
                (
                    'relation_type_parent_id',
                    Subquery(self.vbc.filter(pk=OuterRef('relation_type_id')).values('top_relation_id'))
                ),
                (
                    'relation_type_parent',
                    Concat(
                        F('content_type'),
                        Value('-'),
                        ExpressionWrapper(F('relation_type_parent_id'), output_field=CharField()),
                    )
                ),
                (   # this overwrites the 'ner_tag' annotation from Corpus_Relations
                    'ner_tag',
                    F('relation_type_parent')
                ),
            ] + annotate
        )

class Corpus4CRel(Corpus_CoarseRelations, Corpus4Per):
    def __init__(self):
        super().__init__()
