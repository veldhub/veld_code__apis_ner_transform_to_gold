import logging
import random
import spacy
from spacy.gold import GoldParse, docs_to_json, spans_from_biluo_tags, biluo_tags_from_offsets

class NamedEntity:
    """Represents a named entity as highlighted in a text."""

    def __init__(self, start, end, label, text=None, context=None, annotation_ids=None):
        """
        Parameters:
        ----------
        start : int
            Offset relative to `context`.
        end : int
            Offset relative to `context`.
        label : TODO
            The type of entity.
        text : str
            The 'name' of the named entity, i.e. what was highlighted.
        context : str
            The sentence in which the named entity appears.
        annotation_ids : list of int
            The IDs of all overlapping annotations that were merged to create this named entity.
        """
        self.logger = logging.getLogger(__name__)
        if not start < end:
            self.logger.error(f"Failed to create an entity from index {start} to {end}. Further information: label `{label}`, text `{text}`, context `{context}`.")
            raise AssertionError("Start index must be less than end index.")
        self.start = start
        self.end = end
        self.label = label
        self.text = text
        self.context = context
        self.annotation_ids = annotation_ids

    @classmethod
    def from_tuple(cls, tup):
        assert type(tup) is tuple
        assert len(tup) == 3
        assert type(tup[0]) is int
        assert type(tup[1]) is int
        assert tup[0] < tup[1]
        assert type(tup[2]) is str
        return cls(tup[0], tup[1], tup[2])

    @classmethod
    def from_span(cls, span, doc):
        assert type(span) is spacy.tokens.span.Span
        assert type(doc) is spacy.tokens.doc.Doc
        start_position = sum([ len(token.text_with_ws) for token in doc[0:span.start] ])
        end_position = start_position + len(span.text_with_ws)
        return cls(start_position, end_position, span.label_)

    def is_before(self, ent2): # Returns True iff ent1 is entirely to the left of ent2
        return self.end < ent2.start

    def is_after(self, ent2): # Returns True iff ent1 is entirely to the right of ent2
        return self.start > ent2.end

    def overlaps_with(self, ent2): # Returns True iff ent1 and ent2 overlap in their offsets
        if self.start < ent2.start:
            return ent2.start <= self.end
        elif self.start == ent2.start:
            return True
        elif ent2.start < self.start:
            return self.start <= ent2.end
        else:
            assert False

    def overlaps_with_in_whitespace(self, ent2, text): # Returns True iff the overlap between ent1 and ent2 is only whitespace
        is_whitespace = True
        if self.start != ent2.start:
            is_whitespace &= text[min(self.start,ent2.start) : max(self.start,ent2.start)].isspace()
        if self.end != ent2.end:
            is_whitespace &= text[min(self.end,ent2.end) : max(self.end,ent2.end)].isspace()
        return is_whitespace

    def to_tuple(self):
        return (self.start, self.end, self.label)

    def to_dict(self):
        return {"start": self.start, "end": self.end, "label": self.label}

    def displacy(self):
        assert self.context is not None
        spacy.displacy.render([{
            "text": self.context,
            "ents": [ self.to_dict() ],
        }], manual=True, style="ent", jupyter=True)

Entity = NamedEntity # for backward compatibility

class Datapoint:
    def __init__(self, sentence, entities, doc=None, goldparse=None):
        self.sentence = sentence
        self.entities = entities
        self.doc = doc
        self.goldparse = goldparse

    def __getitem__(self, index):
        if index == 0:
            return self.sentence
        elif index == 1:
            return self.entities
        elif index == 2:
            return self.doc
        elif index == 3:
            return self.goldparse
        else:
            raise IndexError("Datapoint supports only indices 0-3.")

    def tokenize(self, tokenizer_model):
        self.doc = tokenizer_model.tokenize(self.sentence)

    def sentencize(self, sentence_splitter_model):
        self.doc = sentence_splitter_model.sentencize(self.sentence)

    def parse_gold(self):
        self.goldparse = GoldParse(self.doc, entities=[ e.to_tuple() for e in self.entities ])

    def displacy_ents(self):
        # see here for documentation:
        # https://spacy.io/usage/visualizers#manual-usage
        # https://spacy.io/api/top-level#displacy_options-ent
        opt = {}
        opt['colors'] = { # these are spacy's default colors
            "ORG": "#7aecec",
            "LOC": "#ff9561",
            "PER": "#aa9cfc",
        }
        # use the parent color for all fine-grained labels
        # TODO: this doesn't work
        for e in self.entities:
            coarse_label = e.label.split('-')[0]
            assert coarse_label in opt['colors']
            opt['colors'][e.label] = opt['colors'][coarse_label]
        # opt["ents"] = [ t for t in EntityLabel.all() ]
        spacy.displacy.render([{
            "text": self.sentence,
            "ents": [ e.to_dict() for e in self.entities ],
        }], manual=True, style="ent", jupyter=True, options=opt)

    def displacy_parse(self):
        spacy.displacy.render(self.doc, style="dep", jupyter=True)

    def ents_into_doc(self):
        assert self.doc is not None
        # if not self.doc.is_nered:
        self.doc.ents = spans_from_biluo_tags(
            self.doc,
            biluo_tags_from_offsets(
                self.doc,
                [ e.to_tuple() for e in self.entities ]
            )
        )

class Dataset(list):
    def __init__(self, *args):
        if len(args) > 0:
            arg = args[0]
        else:
            arg = []
        super(Dataset, self).__init__(arg)
        self.path_to_vectors = None
        self.path_to_data = None

    def split(self, ratio=0.8):
        data = list(self) # this creates a deep copy
        random.shuffle(data)
        train_data = Dataset(data[ 0 : int(len(data) * ratio) ])
        eval_data = Dataset(data[ len(train_data) : ])
        return train_data, eval_data

    def to_json(self):
        for d in self:
            d.ents_into_doc()
        return [ docs_to_json([ d.doc for d in self ]) ]
