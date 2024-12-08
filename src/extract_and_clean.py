import copy
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import List

import yaml


# output data
UNCLEANED_FOLDER = "/veld/output/uncleaned/"
CLEANED_FOLDER = "/veld/output/cleaned/"
CLEANED_SIMPLIFIED_FOLDER = "/veld/output/cleaned_simplified/"
OUT_JSON_UNCLEANED_FILE = UNCLEANED_FOLDER + os.getenv("out_json_uncleaned_file")
OUT_JSON_CLEANED_FILE = CLEANED_FOLDER + os.getenv("out_json_cleaned_file") 
OUT_JSON_CLEANED_SIMPLIFIED_FILE = CLEANED_SIMPLIFIED_FOLDER + os.getenv("out_json_cleaned_simplified_file")
OUT_VELD_YAML_UNCLEANED_FILE = UNCLEANED_FOLDER + "veld.yaml"
OUT_VELD_YAML_CLEANED_FILE = CLEANED_FOLDER + "veld.yaml"
OUT_VELD_YAML_CLEANED_SIMPLIFIED_FILE = CLEANED_SIMPLIFIED_FOLDER + "veld.yaml"
OUT_LOG_FILE = "/veld/output/log/" + os.getenv("out_log_file")

# output veld metadata
VELD_DATA_TEMPLATE = {
    "x-veld": {
        "data": {
            "file_type": "json",
            "description": None,
            "contents": ["gold data", "NER gold data", "NLP gold data"],
            "topics": ["NLP", "Named entity recognition"],
            "additional": None,
        }
    }
}
OUT_VELD_DATA_UNCLEANED = copy.deepcopy(VELD_DATA_TEMPLATE)
OUT_VELD_DATA_UNCLEANED["x-veld"]["data"]["description"] = "The original, but united, data " \
    "coming from APIS / ÖBL."
OUT_VELD_DATA_CLEANED = copy.deepcopy(VELD_DATA_TEMPLATE)
OUT_VELD_DATA_CLEANED["x-veld"]["data"]["description"] = "Overlapping entities are removed, index" \
    " offsets corrected, and duplicates removed. Also texts without any entities are removed too," \
    " since it's not known if they don't contain any entities (which often is not true; quite a" \
    " few of them contain entities) or if the annotators simply didn't go through them (which is" \
    " more likely, hence they were removed). In the original uncleaned data, some entity types" \
    " are suffixed with numbers (e.g. `PER-1337`). These were used for identifying entities in a" \
    " project context, but are probably of less use for NER NLP training. This dataset keeps the" \
    " identifiers."
OUT_VELD_DATA_CLEANED_SIMPLIFIED = copy.deepcopy(VELD_DATA_TEMPLATE)
OUT_VELD_DATA_CLEANED_SIMPLIFIED["x-veld"]["data"]["description"] = "Same as the cleaned data," \
    " but with simplified entities (e.g. `PER` instead of `PER-1337`). Probably it's best to use" \
    " this data set for NER training."


@dataclass
class TextEntCarrier:
    """
    Helper class to encapsulate texts and their NER tags. All the conversion processes use this as
    transformation target.
    """
    
    @dataclass
    class EntityMarker:
        index_beginning: int
        index_end: int
        entity_type: str
    
    text_raw: str
    entity_marker_list: List[EntityMarker]
    
    def to_dict(self):
        return {
            "text_raw": self.text_raw,
            "entities": [
                (em.index_beginning, em.index_end, em.entity_type)
                for em in self.entity_marker_list
            ]
        }
    
    def __str__(self):
        em_str_list = []
        for em in self.entity_marker_list:
            em_str_list.append(
                f"({em.index_beginning}, {em.index_end}, '{em.entity_type}'"
                f", '{self.text_raw[em.index_beginning:em.index_end]})'"
            )
        return (f"text_raw: '{self.text_raw}', entity_marker_list: {em_str_list}")
    
    def __repr__(self):
        return self.__str__()


def print_and_log(msg):
    print(msg)
    logging.debug(msg)
    

def convert_text_entities_tuple_format(text_entities_tuple_list):
    """
    Some data sets are simple tuples, containing the text as string, and the NER tags as tuples
    within a dictionary. This function converts that kind of data.
    """
    text_ent_carrier_list_tmp = []
    for text_entities_tuple in text_entities_tuple_list:
        entity_marker_list_tmp = []
        for entities_tuple in text_entities_tuple[1]["entities"]:
            entity_marker_list_tmp.append(
                TextEntCarrier.EntityMarker(
                    index_beginning=entities_tuple[0],
                    index_end=entities_tuple[1],
                    entity_type=entities_tuple[2],
                )
            )
        text_ent_carrier_list_tmp.append(
            TextEntCarrier(
                text_raw=text_entities_tuple[0], entity_marker_list=entity_marker_list_tmp
            )
        )
        
    return text_ent_carrier_list_tmp


def convert_text_entities_class_format(text_entities_class_list):
    """
    Some data sets are persisted as pickles of custom data classes defined in the spacy-ner repo.
    This function converts that kind of data.
    """
    text_ent_carrier_list_tmp = []
    for text_entities_class in text_entities_class_list:
        entity_marker_list_tmp = []
        for entities_class in text_entities_class[1]:
            entity_marker_list_tmp.append(
                TextEntCarrier.EntityMarker(
                    index_beginning=entities_class.start,
                    index_end=entities_class.end,
                    entity_type=entities_class.label,
                )
            )
        text_ent_carrier_list_tmp.append(
            TextEntCarrier(
                text_raw=text_entities_class[0], entity_marker_list=entity_marker_list_tmp
            )
        )
    
    return text_ent_carrier_list_tmp


def convert_txt_data(file_path):

    def read_data_from_txt(mypath):
        """
        copied from:
        https://gitlab.oeaw.ac.at/acdh-ch/apis/spacy-ner/-/blob/8e75d3561e617f1bd135d4c06fbb982285f6f544/notebooks/NER%20Place%20Institution.ipynb
        """
        mydata = []
        with open(mypath, "r") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                while lines[i].isspace():
                    i += 1
                # we found a non-empty line to use for t
                t = lines[i].strip()
                i += 1
                while lines[i].isspace():
                    i += 1
                # we found a non-empty line to use for e if possible, else for t
                e = None
                while e == None:
                    try:
                        e = eval(lines[i])
                    except SyntaxError:
                        t += lines[i].strip()
                        i += 1
                        while lines[i].isspace():
                            i += 1
                        # we found a non-empty line to try to use for e again
                i += 1
                mydata.append((t, e, None, None))
        return mydata
    
    print_and_log(f"convert_txt_data: {file_path}")
    return convert_text_entities_tuple_format(read_data_from_txt(file_path))


def convert_pickled_ner_tuples(file_path):
    print_and_log(f"convert_pickled_ner_tuples: {file_path}")
    return convert_text_entities_tuple_format(pickle.load(open(file_path, "rb")))


def convert_pickled_ner_classes(file_path):
    print_and_log(f"convert_pickled_ner_classes: {file_path}")
    return convert_text_entities_class_format(pickle.load(open(file_path, "rb")))

    
def evaluate_json_data(file_path, text_ent_carrier_list):
    """
    This function counts the uniqueness of the single json data set, to see if the json data is
    worth converting. And it's not. The json data  contains no unique texts. So this function is
    not called anymore, but left here for potential re-validation.
    """
    count_found = 0
    count_not_found = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for paragraph in json.load(f)["paragraphs"]:
            found_this = False
            for tec in text_ent_carrier_list:
                if paragraph["raw"] in tec.text_raw:
                    count_found += 1
                    found_this = True
            if not found_this:
                count_not_found += 1
        print_and_log(f"count_found: {count_found}, count_not_found: {count_not_found}")


def extract_all() -> List[TextEntCarrier]:
    text_ent_carrier_list: List[TextEntCarrier] = []
    text_ent_carrier_list.extend(
        convert_txt_data("/veld/input/ner_apis_2019-12-03_23:32:24/corpus/trainset.txt")
    )
    text_ent_carrier_list.extend(
        convert_txt_data("/veld/input/ner_apis_2019-12-03_23:32:24/corpus/evalset.txt")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-02_12:34:48/corpus/trainset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-02_12:34:48/corpus/evalset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-29_13:19:53/corpus/trainset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-29_13:19:53/corpus/evalset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-07_15:00:35/corpus/trainset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-07_15:00:35/corpus/evalset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-16_14:21:46/corpus/trainset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-16_14:21:46/corpus/evalset.pickle")
    )
    text_ent_carrier_list.extend(
        convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-30_11:24:09/corpus/trainset.pickle")
    )
    # evaluate_json_data(
    #     "/veld/input/ner_apis_2020-04-30_11:24:09/corpus/evalset.json", text_ent_carrier_list
    # )
    return text_ent_carrier_list
    
    
def deduplicate(text_ent_carrier_list: List[TextEntCarrier]) -> List[TextEntCarrier]:
    """
    The parsed data from the steps beforehand contain multiple redundancies. Both in texts and the
    entities some texts are assigned on. This function removes all of them.
    """

    # collect all NER data for each text
    tec_dict = {}
    for tec in text_ent_carrier_list:
        tec_pre = tec_dict.get(tec.text_raw, None)

        # If some text with entities already exists before it, merge the two
        if tec_pre is not None and tec_pre.entity_marker_list != tec.entity_marker_list:
            tec.entity_marker_list.extend(tec_pre.entity_marker_list)

        tec_dict[tec.text_raw] = tec
    
    # go through all NER data and remove duplicates, store into new list
    text_ent_carrier_list_new = []
    for tec in tec_dict.values():

        # create new object
        tec_new = TextEntCarrier(text_raw=tec.text_raw, entity_marker_list=[])
        
        # Create a temporary set to guarantee the uniqueness of tag, beginning, end
        em_set_tmp = set()
        for em in tec.entity_marker_list:
            em_tuple = (em.index_beginning, em.index_end, em.entity_type)
            em_set_tmp.add(em_tuple)
        em_list_tmp = list(em_set_tmp)

        # sort by all ner data, to make the output data reliably consistent.
        em_list_tmp.sort(key=lambda x: x[2])
        em_list_tmp.sort(key=lambda x: x[1])
        em_list_tmp.sort(key=lambda x: x[0])
        for em in em_list_tmp:
            tec_new.entity_marker_list.append(
                TextEntCarrier.EntityMarker(
                    index_beginning=em[0], index_end=em[1], entity_type=em[2],
                )
            )
            
        text_ent_carrier_list_new.append(tec_new)
        
    return text_ent_carrier_list_new


def remove_empty(text_ent_carrier_list: List[TextEntCarrier]) -> List[TextEntCarrier]:
    text_ent_carrier_list_new = []
    for tec in text_ent_carrier_list:
        if tec.entity_marker_list != []:
            text_ent_carrier_list_new.append(tec)
    return text_ent_carrier_list_new


def remove_ner_noise(text_ent_carrier_list: List[TextEntCarrier]) -> List[TextEntCarrier]:
    """Removes suspected noise like 'PER-1337' from NER tags."""
    for tec in text_ent_carrier_list:
        for em in tec.entity_marker_list:
            em.entity_type = re.sub(r"-[0-9]*$", "", em.entity_type)
    return text_ent_carrier_list


# def fix_borders(ent_list, text):
def fix_borders(text_ent_carrier_list: List[TextEntCarrier]) -> List[TextEntCarrier]:
    
    def fix_char_index(text, i, i_is_start):
        """
        Takes the start or end index of a substring and moves it to the next valid character. Plenty
        of data has their indices offset by a few positions, so this needs to be corrected.
        
        Since the potentially wrong character of a given index can be either an invalid character (
        space, punctuation, etc.) or a valid character, this algorithm searches for the nearest
        border of character pairs. A border is defined as a difference between character pairs in
        one being valid and the other not.
        
        There is also a small bias introduced depending on the index being the start or end of a
        substring. If the index is at the start, then the search goes first to the left and then to
        the right, and then increases the search radius by a step of 1 in both directions, again
        searching first to the left. If the index is at the end, this is reversed.
        
        Also an explicit exception rule is introduced at the end checking for periods. Since a lot
        of periods are used as abbreviations (e.g. "N.Ö.") and this algorithm here interprets
        periods as invalid characters, a check at the end would correct this if the period is not
        at the end of the whole string, which would indicate a sentence boundary.
        """
        
        def is_char(s):
            return re.compile(r'[\w\d]', re.U).match(s)
        
        def is_not_char(s):
            return not is_char(s)
        
        # The boundary detection benefits from artificial boundaries attached to the start and end
        # so that the edge case of substrings at the start or end don't need to be handled with
        # some dedicated logic. indices i and i_correct later on must be corrected by one step.
        text = " " + text + " "
        i += 1
        # Depending on the index being at the start or end, the boundary must be flipped accordingly
        if i_is_start:
            is_border = lambda a, b: is_not_char(a) and is_char(b)
            i_step = -1
        else:
            is_border = lambda a, b: is_char(a) and is_not_char(b)
            i_step = 1
        i_current = i
        found = False
        while not found:
            i_a = i_current- 1
            i_b = i_current
            if i_a >= 0 and i_b < len(text) and is_border(text[i_a], text[i_b]):
                i_corrected = i_b
                found = True
            else:
                i_current += i_step
                if i_step > 0:
                    i_step += 1
                else:
                    i_step -= 1
                i_step *= -1
        # dedicated logic for the edge case of abbreviations. If a given period is not a sentence
        # boundary, it is included.
        if text[i_corrected] == "." and not re.compile(r'^\. *$', re.U).match(text[i_corrected:]):
            i_corrected += 1
        return i_corrected - 1
    
    def fix_parenthesis(ent, text):
        """
        There are some cases, where both the original data, and the correct one from the functions
        before have lost a parenthesis (e.g. 'Bad) Ischl'). This function checks if a character next
        to the boundaries is a parenthesis, and if it is, then checks if a non-matched counter
        exists in the substring. If so, then it is added.
        """
        i_a = ent[0]
        i_b = ent[1]
        text_sub = text[i_a:i_b]
        ent_new = ent
        if i_a > 0:
            if text[i_a - 1] == "(" and text_sub.count(")") > text_sub.count("("):
                ent_new = [i_a - 1, i_b, ent[2]]
        if i_b < len(text) - 1:
            if text[i_b] == ")" and text_sub.count("(") > text_sub.count(")"):
                ent_new = [i_a, i_b + 1, ent[2]]
        return ent_new
    
    def fix_borders_main(text_ent_carrier_list):
        text_ent_carrier_list_new = []
        for tec in text_ent_carrier_list:
            ent_list = [
                [em.index_beginning, em.index_end, em.entity_type]
                for em in tec.entity_marker_list
            ]
            text = tec.text_raw
            tec_new = TextEntCarrier(text, [])
            for ent in ent_list:
                ent_new = [
                    fix_char_index(text, ent[0], True),
                    fix_char_index(text, ent[1], False),
                    ent[2]
                ]
                ent_new = fix_parenthesis(ent_new, text)
                tec_new.entity_marker_list.append(TextEntCarrier.EntityMarker(
                    index_beginning=ent_new[0], index_end=ent_new[1], entity_type=ent_new[2]
                ))
                if ent != ent_new:
                    print_and_log(
                        f"replaced: {ent}, text: {text[ent[0]:ent[1]].__repr__()},"
                        f" with: {ent_new}, text: {text[ent_new[0]:ent_new[1]].__repr__()}"
                    )
                else:
                    print_and_log(f"nothing replaced: {ent}, {text[ent[0]:ent[1]].__repr__()}")
                assert not (
                    text[ent_new[0]:ent_new[1]].startswith(" ")
                    or text[ent_new[0]:ent_new[1]].endswith(" ")
                )
            text_ent_carrier_list_new.append(tec_new)
        return text_ent_carrier_list_new
    
    return fix_borders_main(text_ent_carrier_list)


def write_to_file(
    text_ent_carrier_list: List[TextEntCarrier], 
    output_data_path, 
    output_veld_data,
    output_veld_data_path,
):
    text_ent_dict_list = [tec.to_dict() for tec in text_ent_carrier_list]

    # calculate stats for metadata
    stats_entities = {}
    stats_total_count = 0
    for text_ent_dict in text_ent_dict_list:
        for ent in text_ent_dict["entities"]:
            ent_count = stats_entities.get(ent[2], 0)
            stats_entities[ent[2]] = ent_count + 1
            stats_total_count += 1
    stats_entities = {
        "total count of entities": stats_total_count,
        "individual count of entities": stats_entities
    }
    output_veld_data["x-veld"]["data"]["additional"] = stats_entities

    # write data and veld metadata
    with open(output_veld_data_path, "w") as f:
        yaml.dump(output_veld_data, f, sort_keys=False, allow_unicode=True)
    with open(output_data_path, "w") as f:
        json.dump(text_ent_dict_list, f, ensure_ascii=False, indent=2)
        

def main():
    
    # extraction
    print_and_log("Starting extraction.")
    text_ent_carrier_list = extract_all()
    print_and_log(f"Done with extraction. Length of raw converted data: "
        f"{len(text_ent_carrier_list)}"
    )

    # deduplication
    print_and_log("Starting deduplication.")
    text_ent_carrier_list = deduplicate(text_ent_carrier_list)
    print_and_log(f"Done with deduplication. Length of deduplicated data: "
        f"{len(text_ent_carrier_list)}"
    )
    write_to_file(
        text_ent_carrier_list,
        OUT_JSON_UNCLEANED_FILE,
        OUT_VELD_DATA_UNCLEANED,
        OUT_VELD_YAML_UNCLEANED_FILE
    )

    # clean
    print_and_log("Removing empty entity data items.")
    text_ent_carrier_list = remove_empty(text_ent_carrier_list)
    print_and_log(f"Done with removing empty entity data items. Length of cleaned data: "
        f"{len(text_ent_carrier_list)}"
    )
    print_and_log("Fixing borders.")
    text_ent_carrier_list = fix_borders(text_ent_carrier_list)
    text_ent_carrier_list = deduplicate(text_ent_carrier_list)
    write_to_file(
        text_ent_carrier_list,
        OUT_JSON_CLEANED_FILE,
        OUT_VELD_DATA_CLEANED,
        OUT_VELD_YAML_CLEANED_FILE
    )

    # simplify by removing ner noise
    print_and_log("Starting removal of noise in NER tags.")
    text_ent_carrier_list = remove_ner_noise(text_ent_carrier_list)
    print_and_log("Done with noise removal.")
    print_and_log("Starting deduplication again for denoised data.")
    text_ent_carrier_list = deduplicate(text_ent_carrier_list)
    print_and_log("Done with deduplication.")
    write_to_file(
        text_ent_carrier_list,
        OUT_JSON_CLEANED_SIMPLIFIED_FILE,
        OUT_VELD_DATA_CLEANED_SIMPLIFIED,
        OUT_VELD_YAML_CLEANED_SIMPLIFIED_FILE
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=OUT_LOG_FILE,
        filemode='w',
        level=logging.DEBUG,
        format='%(message)s',
    )
    main()

