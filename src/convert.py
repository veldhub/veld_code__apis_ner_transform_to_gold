import json
import pickle
import re
from dataclasses import dataclass
from typing import List


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
    
    print(f"convert_txt_data: {file_path}")
    return convert_text_entities_tuple_format(read_data_from_txt(file_path))


def convert_pickled_ner_tuples(file_path):
    print(f"convert_pickled_ner_tuples: {file_path}")
    return convert_text_entities_tuple_format(pickle.load(open(file_path, "rb")))


def convert_pickled_ner_classes(file_path):
    print(f"convert_pickled_ner_classes: {file_path}")
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
        
        print(f"count_found: {count_found}, count_not_found: {count_not_found}")


def convert_all() -> List[TextEntCarrier]:
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


def remove_ner_noise(text_ent_carrier_list: List[TextEntCarrier]) -> List[TextEntCarrier]:
    """Removes suspected noise like 'PER-1337' from NER tags."""
    for tec in text_ent_carrier_list:
        for em in tec.entity_marker_list:
            em.entity_type = re.sub(r"-[0-9]*$", "", em.entity_type)
    
    return text_ent_carrier_list


def write_to_file(text_ent_carrier_list: List[TextEntCarrier], output_path):
    text_ent_dict_list = [tec.to_dict() for tec in text_ent_carrier_list]
    with open(output_path, "w") as f:
        json.dump(text_ent_dict_list, f, indent=2)
        

def main():
    # conversion
    print("Starting conversion.")
    text_ent_carrier_list = convert_all()
    print(f"Done with conversion. Length of raw converted data: {len(text_ent_carrier_list)}")
    # deduplication
    print("Starting deduplication.")
    text_ent_carrier_list = deduplicate(text_ent_carrier_list)
    print(f"Done with deduplication. Length of deduplicated data: {len(text_ent_carrier_list)}")
    write_to_file(text_ent_carrier_list, "/veld/output/json/apis_ner__full_entities.json")
    # removing noise
    print("Starting removal of noise in NER tags.")
    text_ent_carrier_list = remove_ner_noise(text_ent_carrier_list)
    print("Done with noise removal.")
    # deduplication of  denoised data
    print("Starting deduplication again for denoised data.")
    text_ent_carrier_list = deduplicate(text_ent_carrier_list)
    print(f"Done with deduplication.")
    write_to_file(text_ent_carrier_list, "/veld/output/json/apis_ner__simplified_entities.json")
    print(f"All Done and persisted to '/veld/output/json/'.")


main()
