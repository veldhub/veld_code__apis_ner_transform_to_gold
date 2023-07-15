import spacy
import pickle
import json
from dataclasses import dataclass
from typing import List


@dataclass
class TextEntitiesCarrier:
    @dataclass
    class EntityMarker:
        index_beginning: int
        index_end: int
        entity_type: str
    
    text_raw: str
    entity_marker_list: List[EntityMarker]
    
    def __str__(self):
        em_str_list = []
        for em in self.entity_marker_list:
            em_str_list.append(
                f"({em.index_beginning}, {em.index_end}, '{em.entity_type}'"
                f", '{self.text_raw[em.index_beginning:em.index_end]})'"
            )
        return (
            f"text_raw: {self.text_raw}, entity_marker_list: {em_str_list}"
        )
    
    def __repr__(self):
        return self.__str__()


def read_data_from_pickle(file_path):
    def pickle_stats(pickle_data):
        pickle_stats = {}
        for d in pickle_data:
            l = len(d[1]["entities"])
            c = pickle_stats.get(l, 0)
            c += 1
            pickle_stats[l] = c
        print(f"pickle_stats: {pickle_stats}")
    
    return pickle.load(open(file_path, "rb"))


def convert_text_entities_tuple_format(text_entities_tuple_list):
    text_entities_carrier_list_tmp = []
    for text_entities_tuple in text_entities_tuple_list:
        entity_marker_list_tmp = []
        for entities_tuple in text_entities_tuple[1]["entities"]:
            entity_marker_list_tmp.append(
                TextEntitiesCarrier.EntityMarker(
                    index_beginning=entities_tuple[0],
                    index_end=entities_tuple[1],
                    entity_type=entities_tuple[2],
                )
            )
        
        text_entities_carrier_list_tmp.append(
            TextEntitiesCarrier(
                text_raw=text_entities_tuple[0],
                entity_marker_list=entity_marker_list_tmp
            )
        )
        
    return text_entities_carrier_list_tmp


def convert_text_entities_class_format(text_entities_class_list):
    text_entities_carrier_list_tmp = []
    for text_entities_class in text_entities_class_list:
        entity_marker_list_tmp = []
        for entities_class in text_entities_class[1]:
            entity_marker_list_tmp.append(
                TextEntitiesCarrier.EntityMarker(
                    index_beginning=entities_class.start,
                    index_end=entities_class.end,
                    entity_type=entities_class.label,
                )
            )
        
        text_entities_carrier_list_tmp.append(
            TextEntitiesCarrier(
                text_raw=text_entities_class[0],
                entity_marker_list=entity_marker_list_tmp
            )
        )
    
    return text_entities_carrier_list_tmp


def convert_txt_data(file_path):
    def read_data_from_txt(mypath):
        """
        copied from: https://gitlab.oeaw.ac.at/acdh-ch/apis/spacy-ner/-/blob/8e75d3561e617f1bd135d4c06fbb982285f6f544/notebooks/NER%20Place%20Institution.ipynb
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
    
    print(f"convert_txt_data: {file_path}")
    return convert_text_entities_tuple_format(read_data_from_txt(file_path))


def convert_pickled_ner_tuples(file_path):
    print(f"convert_pickled_ner_tuples: {file_path}")
    return convert_text_entities_tuple_format(read_data_from_pickle(file_path))


def convert_pickled_ner_classes(file_path):
    print(f"convert_pickled_ner_classes: {file_path}")
    return convert_text_entities_class_format(read_data_from_pickle(file_path))

    
def evaluate_json_data(file_path, text_entities_carrier_list):
    """
    # summary:
    This function counts the uniqueness of the json data set, to see if the json data is worth
    converting.
    
    **spoiler: it's not. It only contains 8 unique texts (and these might be even miscounted given
    the naive implementation of comparison). So not worth converting.**
    """
    count_found = 0
    count_not_found = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for paragraph in json.load(f)["paragraphs"]:
            found_this = False
            for tec in text_entities_carrier_list:
                if paragraph["raw"] in tec.text_raw:
                    count_found += 1
                    found_this = True
                    
            if not found_this:
                count_not_found += 1
                
        print(f"count_found: {count_found}, count_not_found: {count_not_found}")


def convert_all() -> List[TextEntitiesCarrier]:
    text_entities_carrier_list: List[TextEntitiesCarrier] = []
    text_entities_carrier_list.extend(
        convert_txt_data(
            "/veld/input/ner_apis_2019-12-03_23:32:24/corpus/trainset.txt"
        )
    )
    text_entities_carrier_list.extend(
        convert_txt_data(
            "/veld/input/ner_apis_2019-12-03_23:32:24/corpus/evalset.txt"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_tuples(
            "/veld/input/ner_apis_2020-01-02_12:34:48/corpus/trainset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_tuples(
            "/veld/input/ner_apis_2020-01-02_12:34:48/corpus/evalset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_tuples(
            "/veld/input/ner_apis_2020-01-29_13:19:53/corpus/trainset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_tuples(
            "/veld/input/ner_apis_2020-01-29_13:19:53/corpus/evalset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_classes(
            "/veld/input/ner_apis_2020-04-07_15:00:35/corpus/trainset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_classes(
            "/veld/input/ner_apis_2020-04-07_15:00:35/corpus/evalset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_classes(
            "/veld/input/ner_apis_2020-04-16_14:21:46/corpus/trainset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_classes(
            "/veld/input/ner_apis_2020-04-16_14:21:46/corpus/evalset.pickle"
        )
    )
    text_entities_carrier_list.extend(
        convert_pickled_ner_classes(
            "/veld/input/ner_apis_2020-04-30_11:24:09/corpus/trainset.pickle"
        )
    )
    # no conversion, just an evaluation of potential redundancy with regard to other data sets
    evaluate_json_data(
        "/veld/input/ner_apis_2020-04-30_11:24:09/corpus/evalset.json",
        text_entities_carrier_list
    )
    print(f"done. length of raw covnerted data: {len(text_entities_carrier_list)}")
    return text_entities_carrier_list
    

def deduplicate(text_entities_carrier_list): #TODO: validate this really properly
    tec_dict = {}
    for tec in text_entities_carrier_list:
        tec_pre = tec_dict.get(tec.text_raw, None)
        if tec_pre is not None and tec_pre.entity_marker_list != tec.entity_marker_list:
            em_set_tmp = set()
            for em in tec_pre.entity_marker_list:
                em_tuple = (em.index_beginning, em.index_end, em.entity_type)
                em_set_tmp.add(em_tuple)
            
            for em in tec.entity_marker_list:
                em_tuple = (em.index_beginning, em.index_end, em.entity_type)
                em_set_tmp.add(em_tuple)
            
            em_list = list(em_set_tmp)
            em_list.sort(key=lambda x : x[0])
            tec.entity_marker_list = []
            for em in em_list:
                tec.entity_marker_list.append(
                    TextEntitiesCarrier.EntityMarker(
                        index_beginning=em[0],
                        index_end=em[1],
                        entity_type=em[2],
                    )
                )
                
        tec_dict[tec.text_raw] = tec
    
    text_entities_carrier_list_new = []
    for tec in tec_dict.values():
        text_entities_carrier_list_new.append(tec)
        
    return text_entities_carrier_list_new


def write_to_file(text_entities_carrier_list):
    pass # TODO


def main():
    text_entities_carrier_list = convert_all()
    text_entities_carrier_list = deduplicate(text_entities_carrier_list)
    write_to_file(text_entities_carrier_list)


main()