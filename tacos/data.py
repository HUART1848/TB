import json
import random

class ContextSentence():
    pre: str = ...
    sen: str = ...

    def __init__(self, pre: str, sen: str):
        self.pre = pre
        self.sen = sen

class ComparisonExample():
    src: ContextSentence
    trg_correct: ContextSentence
    trg_incorrect: ContextSentence

    def __init__(self,
                 src: ContextSentence,
                 trg_correct: ContextSentence,
                 trg_incorrect: ContextSentence):
        self.src = src
        self.trg_correct = trg_correct
        self.trg_incorrect = trg_incorrect

class ComparisonTestSet():
    src_lang: str = ...
    trg_lang: str = ...
    examples: list[ComparisonExample] = ...

    def __init__(self, src_lang: str, trg_lang: str):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.examples = list()

    def add_example(self, example: ComparisonExample):
        self.examples.append(example)

    def to_dict(self) -> list[dict]:
        ret = list()

        for example in self.examples:
            ret.append({
                "src_pre": example.src.pre,
                "src_sen": example.src.sen,
                "trg_correct_pre": example.trg_correct.pre,
                "trg_correct_sen": example.trg_correct.sen,
                "trg_incorrect_pre": example.trg_incorrect.pre,
                "trg_incorrect_sen": example.trg_incorrect.sen
            })

        return ret
    
class TestSetLoader():
    def comparison_from_anaphora(self, path: str) -> ComparisonTestSet:
        ...

class DiscourseMTLoader(TestSetLoader):
    SRC_TAG = "src"
    TRG_TAG = "trg"
    CORRECT_TAG = "correct"
    SEMI_CORRECT_TAG = "semi-correct"
    INCORRECT_TAG = "incorrect"

    def comparison_from_anaphora(self, path: str) -> ComparisonTestSet:
        ret = ComparisonTestSet("en", "fr")

        with open(path, "r") as f:
            raw = json.load(f)
            for i in raw.keys():
                src_pre = raw[i][self.SRC_TAG][0]
                src_sen = raw[i][self.SRC_TAG][1]

                for trg in raw[i][self.TRG_TAG]:
                    correct_tag = self.CORRECT_TAG if self.CORRECT_TAG in trg.keys() else self.SEMI_CORRECT_TAG

                    example = ComparisonExample(
                        ContextSentence(src_pre, src_sen),
                        ContextSentence(trg[correct_tag][0], trg[correct_tag][1]),
                        ContextSentence(trg[self.INCORRECT_TAG][0], trg[self.INCORRECT_TAG][1]),
                    )

                    ret.add_example(example)
            
        return ret

    def comparison_from_lexical_choice(self, path: str) -> ComparisonTestSet:
        ret = ComparisonTestSet("en", "fr")

        with open(path, "r") as f:
            raw = json.load(f)
            for i in raw.keys():
                for e in raw[i]["examples"]:
                    src_pre = e[self.SRC_TAG][0]
                    src_sen = e[self.SRC_TAG][1]

                    trg = e[self.TRG_TAG]
                    correct_tag = self.CORRECT_TAG
                    example = ComparisonExample(
                        ContextSentence(src_pre, src_sen),
                        ContextSentence(trg[correct_tag][0], trg[correct_tag][1]),
                        ContextSentence(trg[self.INCORRECT_TAG][0], trg[self.INCORRECT_TAG][1]),
                    )

                    ret.add_example(example)
        return ret
