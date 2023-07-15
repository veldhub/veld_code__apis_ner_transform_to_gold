import spacy
import pickle
import json
from dataclasses import dataclass
from typing import List


iob_list_all = []
nlp = spacy.load('de_core_news_md')


@dataclass
class SqueezedTextWithEntities:
    @dataclass
    class EntitiesWithPositions:
        index_beginning: int
        index_end: int
        entity: str
        text_part: str
    
    text_squeezed: str
    entities_with_positions: List[EntitiesWithPositions]


def convert_to_squeezed_text_with_entities(tuple_with_text_ner_tuples) -> SqueezedTextWithEntities:
    def squeeze_text(text):
        # return text.replace(" ", "").replace("\t", "").replace("\n", "") # TODO
        return text.replace(" ", "") # TODO
    
    def squeeze_text_part_and_indices(text_original, index_beginning_original, index_end_original):
        def calcualate_index_squeezed(text_original, index_original):
            text_original_until_index = text_original[:index_original]
            index_diff = text_original_until_index.count(" ")
            return index_original - index_diff
        
        text_squeezed = squeeze_text(text_original)
        index_beginning_squeezed = calcualate_index_squeezed(
            text_original,
            index_beginning_original
        )
        index_end_squeezed = calcualate_index_squeezed(
            text_original,
            index_end_original
        )
        text_part_original_squeezed = squeeze_text(
            text_original[index_beginning_original:index_end_original]
        )
        text_part_squeezed = text_squeezed[index_beginning_squeezed:index_end_squeezed]
        if text_part_original_squeezed != text_part_squeezed:
            raise Exception
        return text_part_squeezed, index_beginning_squeezed, index_end_squeezed
        
    def main():
        entities_with_positions = []
        text_original = tuple_with_text_ner_tuples[0]
        text_squeezed = squeeze_text(text_original)
        for ner_tuple in tuple_with_text_ner_tuples[1]["entities"]:
            t_p_s, i_b_s, i_e_s = squeeze_text_part_and_indices(
                text_original=text_original,
                index_beginning_original=ner_tuple[0],
                index_end_original=ner_tuple[1],
            )
            entities_with_positions.append(
                SqueezedTextWithEntities.EntitiesWithPositions(
                    index_beginning=i_b_s,
                    index_end=i_e_s,
                    entity=ner_tuple[2],
                    text_part=t_p_s
                )
            )
            
        return SqueezedTextWithEntities(
            text_squeezed=text_squeezed,
            entities_with_positions=entities_with_positions
        )
    
    return main()


def convert_to_iob(text_original: str, stwe: SqueezedTextWithEntities):
    doc = nlp(text_original) # TODO: check if "\t" and "\n" can be excluded
    index_current_beginning = 0
    result = []
    count_ner_token = 0
    count_ner_real = len(stwe.entities_with_positions)
    for token in doc:
        if token.text == " ":
            raise Exception
        # elif token == "\t" # TODO
        index_current_end = index_current_beginning + len(token)
        iob = "O"
        for ewp in stwe.entities_with_positions:
            if (
                ewp.index_beginning <= index_current_beginning < ewp.index_end
                or ewp.index_beginning < index_current_end < ewp.index_end
            ) or (
                index_current_beginning <= ewp.index_beginning < index_current_end
                or index_current_beginning < ewp.index_end < index_current_end
            ):
                iob = "I"
                count_ner_token += 1
                break
        result.append((token.text, iob))
        index_current_beginning = index_current_end
    
    return result, count_ner_token, count_ner_real


# def check_overlap_of_train_eval_data(train_data_text_only, eval_data_text_only):
#     count_train_in_eval = 0
#     count_eval_in_train = 0
#     for i, text in enumerate(train_data_text_only):
#         if text in eval_data_text_only:
#             # print(f"train text in eval data at index: {i}")
#             count_train_in_eval += 1
#
#     for i, text in enumerate(eval_data_text_only):
#         if text in train_data_text_only:
#             # print(f"eval text in train data at index: {i}")
#             count_eval_in_train += 1
#
#     print(f"count of sentences of evaluation data in train data: {count_eval_in_train}")
#     print(f"count of sentences of train data in evaluation data: {count_train_in_eval}")
    
    
def read_pickle_individual(pickle_file_path):
    def pickle_stats(pickle_data):
        pickle_stats = {}
        for d in pickle_data:
            l = len(d[1]["entities"])
            c = pickle_stats.get(l, 0)
            c += 1
            pickle_stats[l] = c
        print(f"pickle_stats: {pickle_stats}")
    
    print(f"loading: {pickle_file_path}")
    return pickle.load(open(pickle_file_path, "rb"))


def convert_pickled_ner_tuples(pickle_file_path):
    converted_data = None
    pickle_data = read_pickle_individual(pickle_file_path)
    
    for data_tuple in pickle_data:
        doc = data_tuple[2]
    
    print("convert_pickled_ner_tuples")
    
    return pickle_data


def convert_pickled_ner_classes(pickle_file_path):
    converted_data = None
    pickle_data = read_pickle_individual(pickle_file_path)
    
    for data_tuple in pickle_data:
        doc = data_tuple[2]
    
    print("convert_pickled_ner_classes")
    
    return pickle_data


def convert_txt_data(data_txt_path):
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
    
    def main():
        iob_list = []
        count_ner_token = 0
        count_ner_real = 0
        for data_txt_row in read_data_from_txt(data_txt_path):
            stwe = convert_to_squeezed_text_with_entities(data_txt_row)
            iob_text, count_ner_token_row, count_ner_real_row = convert_to_iob(data_txt_row[0], stwe)
            iob_list.append(iob_text)
            count_ner_token += count_ner_token_row
            count_ner_real += count_ner_real_row
            
        print(f"number of tokens with entities: {count_ner_token}")
        print(f"number of entities assigned in original data: {count_ner_real}")
        print(f"ratio tokens to real: {round(count_ner_token / count_ner_real, 2)}")
        return iob_list

    main()
    
    
def convert_apis_ner_2020_01_02_until_2020_04_16():
    # convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-02_12:34:48/corpus/trainset.pickle")
    # convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-02_12:34:48/corpus/evalset.pickle")
    # convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-29_13:19:53/corpus/trainset.pickle")
    convert_pickled_ner_tuples("/veld/input/ner_apis_2020-01-29_13:19:53/corpus/evalset.pickle")
    # convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-07_15:00:35/corpus/trainset.pickle")
    # convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-07_15:00:35/corpus/evalset.pickle")
    convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-16_14:21:46/corpus/trainset.pickle")
    # convert_pickled_ner_classes("/veld/input/ner_apis_2020-04-16_14:21:46/corpus/evalset.pickle")


def convert_2020_04_30():
    train_data = convert_pickled_ner_classes(
        f"/veld/input/ner_apis_2020-04-30_11:24:09/corpus/trainset.pickle"
    )
    with open(
        f"/veld/input/ner_apis_2020-04-30_11:24:09/corpus/evalset.json", "r",
        encoding="utf-8"
    ) as f:
        eval_data = json.load(f)["paragraphs"]


def write_iob_list(iob_list, file_path):
    pass


def remove_redundancies_from_iob_list(iob_list):
    return iob_list


def main():
    iob_list_all.extend(
        convert_txt_data("/veld/input/ner_apis_2019-12-03_23:32:24/corpus/trainset.txt")
    )
    iob_list_all.extend(
        convert_txt_data("/veld/input/ner_apis_2019-12-03_23:32:24/corpus/evalset.txt")
    )
    convert_apis_ner_2020_01_02_until_2020_04_16()


main()