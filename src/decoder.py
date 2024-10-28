import re
from collections import defaultdict
from itertools import product

import nltk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .i7h import i18n


class Decoder:
    def __init__(self, model_name: str, lookahead: int = 1):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word_list = self.get_word_list()
        self.i18n_word_list = self.build_i18n_word_list()
        self.i18n_tokenizer = self.build_i18n_tokenizer()
        self.lookahead = lookahead

    def get_word_list(self) -> list[str]:
        nltk.download("words")
        from nltk.corpus import words

        return words.words()

    def build_i18n_word_list(self) -> dict[str, list[str]]:
        word_list = defaultdict(list)
        for word in self.word_list:
            word_list[i18n(word)].append(word)
        return word_list

    def build_i18n_tokenizer(self):
        return nltk.tokenize.NLTKWordTokenizer()

    @staticmethod
    def is_i18n_word(word: str) -> bool:
        return re.match(r"[a-zA-Z]\d+[a-zA-Z]", word) is not None

    def decoder_tokenize(self, text: str) -> list[str]:
        return self.i18n_tokenizer.tokenize(text)

    def decoder_tokenize_with_i18n_flag(self, text: str):
        tokens = self.decoder_tokenize(text)
        flags = [self.is_i18n_word(tokens[i]) for i in range(len(tokens))]
        return tokens, flags

    @staticmethod
    def string_replace(string: str, replacements: list) -> str:
        # replacements: list[(start, end, replacement)]
        replacements = sorted(replacements, key=lambda x: x[0], reverse=True)
        # assert that replacements are not overlapped
        for i in range(1, len(replacements)):
            assert replacements[i][1] <= replacements[i - 1][0]
        for start, end, replacement in replacements:
            string = string[:start] + replacement + string[end:]
        return string

    def get_loss(self, string: str) -> float:
        inputs = self.tokenizer(string, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"]
        outputs = self.model(**inputs)
        return outputs.loss.item()

    def decode(self, lookahead_string: str, verbose: bool = False) -> list[str]:
        decoded_string = ""
        while lookahead_string:
            if verbose:
                print(f"Decoded: {decoded_string} <-> Remaining: {lookahead_string}")
            lookahead_i18n_tokens = []
            for m in re.finditer(r"[a-zA-Z]\d+[a-zA-Z]", lookahead_string):
                n_lookahead = self.lookahead
                if len(decoded_string) == 0:
                    n_lookahead = self.lookahead + 1
                if len(lookahead_i18n_tokens) >= n_lookahead:
                    break
                if m.group() in self.i18n_word_list:
                    lookahead_i18n_tokens.append((m.group(), m.start(), m.end()))
            substring = lookahead_string[: lookahead_i18n_tokens[-1][2]]
            lookahead_string = lookahead_string[lookahead_i18n_tokens[-1][2] :]

            token_candidates = []
            for i18n_token in lookahead_i18n_tokens:
                token_candidates.append(self.i18n_word_list[i18n_token[0]])
            string_combs = []
            for comb in product(*token_candidates):
                comb_tokens = []
                for ti, token in enumerate(comb):
                    comb_tokens.append(
                        (
                            lookahead_i18n_tokens[ti][1],
                            lookahead_i18n_tokens[ti][2],
                            token,
                        )
                    )
                string_combs.append(self.string_replace(substring, comb_tokens))
            string_comb_with_loss = []
            pbar = string_combs
            if verbose:
                pbar = tqdm(string_combs)
            for string_comb in pbar:
                candidate_string = decoded_string + string_comb
                loss = self.get_loss(candidate_string)
                string_comb_with_loss.append((string_comb, loss))
            string_comb_with_loss.sort(key=lambda x: x[1])
            decoded_string += string_comb_with_loss[0][0]
        return decoded_string


if __name__ == "__main__":
    decoder = Decoder("HuggingFaceTB/SmolLM-135M", lookahead=1)
    # tokens = decoder.decoder_tokenize(
    #     "A3e w1s b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k, a1d of h4g n5g to do:  o2e or t3e s1e h1d"
    # )
    # print(tokens)
    # print(decoder.i18n_word_list[tokens[0]])
    # print(decoder.i18n_word_list[tokens[2]])

    """
    Original: Alice was beginning to get very tired of sitting by her sister on the bank
    Lookahead: 1, Time: 39.99s, Result: Annie was bemoaning to get very tired of sobbing by her sister on the back
    """

    import time

    i18n_string = "A3e w1s b7g to g1t v2y t3d of s5g by h1r s4r on t1e b2k"
    for lookahead in range(1, 3):
        decoder.lookahead = lookahead
        start = time.time()
        res = decoder.decode(i18n_string, verbose=True)
        period = time.time() - start
        print(f"Lookahead: {lookahead}, Time: {period:.2f}s, Result: {res}")
