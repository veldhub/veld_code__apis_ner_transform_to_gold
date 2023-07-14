import logging
import os
from pathlib import Path
import pickle
import shutil
import spacy
import subprocess
import time

class GloveVectors:
    def __init__(self, path_glove, path_output, vector_size=300, vector_format="d"):
        self.path_to_glove_installation = path_glove

        self.vector_size = vector_size
        self.vector_format = vector_format

        Path(path_output).mkdir(parents=True, exist_ok=True)
        self.path_to_corpus = f"{path_output}/corpus.txt"
        self.path_to_vocab = f"{path_output}/vocab.txt"
        self.path_to_cooccur = f"{path_output}/cooccurrences.bin"
        self.path_to_cooccur_shuffle = f"{path_output}/cooccurrences.shuf.bin"
        self.path_to_vectors = f"{path_output}/vectors.{self.vector_size}.{self.vector_format}"
        self.path_to_vectors_txt = f"{self.path_to_vectors}.txt"
        self.path_to_vectors_bin = f"{self.path_to_vectors}.bin"

        self.logger = logging.getLogger(__name__)

    def install_glove(self):
        shutil.rmtree(self.path_to_glove_installation)
        p = subprocess.Popen(["git", "clone", "http://github.com/stanfordnlp/glove", self.path_to_glove_installation])
        p.wait()
        assert p.returncode == 0
        p = subprocess.Popen(["cd", self.path_to_glove_installation])
        p.wait()
        p = subprocess.Popen(["make"])
        p.wait()
        assert p.returncode == 0

    def load_corpus(self, dataset, tokenizer):
        corpus = []
        for datapoint in dataset:
            datapoint.tokenize(tokenizer)
            sentence_formatted = ' '.join([ token.text for token in datapoint.doc if not token.is_punct ])
            corpus.append(f"{sentence_formatted}\n")
        with open(self.path_to_corpus, "w") as f:
            f.writelines(corpus)

    def load_corpus_from_file(self, path, tokenizer):
        self.load_corpus(pickle.load(open(path, "rb")), tokenizer)

    def run(self):
        p = subprocess.Popen(
            [
                f"{self.path_to_glove_installation}/build/vocab_count",
                "-verbose", "2"
            ],
            stdin=open(self.path_to_corpus, "r"),
            stdout=open(self.path_to_vocab, "w")
        )
        p.wait()
        assert p.returncode == 0
        p = subprocess.Popen(
            [
                f"{self.path_to_glove_installation}/build/cooccur",
                "-verbose", "2",
                "-symmetric", "1",
                "-window-size", "15",
                "-vocab-file", self.path_to_vocab,
                "-overflow-file", "tempoverflow"
            ],
            stdin=open(self.path_to_corpus, "r"),
            stdout=open(self.path_to_cooccur, "wb")
        )
        p.wait()
        assert p.returncode == 0
        p = subprocess.Popen(
            [
                f"{self.path_to_glove_installation}/build/shuffle",
                "-verbose", "2"
            ],
            stdin=open(self.path_to_cooccur, "rb"),
            stdout=open(self.path_to_cooccur_shuffle, "wb")
        )
        p.wait()
        assert p.returncode == 0
        subprocess.Popen(
            [
                f"{self.path_to_glove_installation}/build/glove",
                "-write-header", "1",
                "-binary", "2",
                "-input-file", self.path_to_cooccur_shuffle,
                "-vocab-file", self.path_to_vocab,
                "-save-file", self.path_to_vectors,
                "-vector-size", str(self.vector_size)
            ]
        )
        p.wait()
        assert p.returncode == 0

        if not os.path.exists(self.path_to_vectors_txt):
            time.sleep(5)
        assert os.path.exists(self.path_to_vectors_txt)

        p = subprocess.Popen(
            [
                "grep",
                "--count",
                "^<unk>",
                self.path_to_vectors_txt
            ],
            stderr=subprocess.PIPE
        )
        p.wait()
        if p.returncode == 2: # 2 would be an error
            self.logger.error(p.stderr.read().decode("utf-8"))
        elif p.returncode == 0: # at least one line was selected
            self.logger.debug("A vector for '<unk>' has been calculated.")
            # the word '<unk>' has appeared, so we need to remove this line from self.path_to_vectors.txt
            tmp = f"{self.path_to_vectors_txt}.tmp"
            orig = f"{self.path_to_vectors_txt}.orig"
            p = subprocess.Popen(
                [
                    "grep",
                    "--invert-match",
                    "^<unk>",
                    self.path_to_vectors_txt
                ],
                stdout=open(tmp, "w"))
            p.wait()
            assert p.returncode != 2
            os.rename(self.path_to_vectors_txt, orig)
            os.rename(tmp, self.path_to_vectors_txt)
            self.logger.info(f"The vector for '<unk>' has been removed from the file '{self.path_to_vectors_txt}'. The original file is available as '{orig}'.")
        else: # == 1
            self.logger.debug("There was no vector for '<unk>'.")

    def save_as_spacy_model(self, path):
        # I'm using the spacy CLI 'init-model' command to initialize a model based on these vectors.
        # Not sure if this could be done in a more simple way in python directly.
        tmp = "/tmp/spacy_init_model"
        p = subprocess.Popen(
            [
                "python",
                "-m", "spacy", "init-model",
                "de",
                tmp,
                "--vectors-loc", self.path_to_vectors_txt
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p.wait()
        self.logger.debug(p.stdout.read().decode("utf-8"))
        self.logger.error(p.stderr.read().decode('utf-8'))
        self.logger.debug(p.returncode)
        assert p.returncode == 0
        self.logger.debug(p.stdout.read().decode("utf-8"))
        nlp = spacy.load(tmp)
        nlp.vocab.vectors.to_disk(path)
        self.logger.info(f"Saved a spacy model with vectors of size {nlp.vocab.vectors.shape}.")
        shutil.rmtree(tmp)
