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


    def to_choice_prompt(self, shuffle=True) -> tuple[int, str]:
        """
        Returns a formatted prompt with the position of the correct sentence (1 or 2)
        """
        ret = f"""
        You will decide which of two translations from English to French is the most correct one.
        The context of the source sentence is '{self.src.pre}'
        The source sentence is '{self.src.sen}'
        The context translated in French is '{self.trg_correct.pre}'
        Which of the following translations is correct? Answer only with '1' or '2'"""

        order = random.randint(1, 2) if shuffle else 1
        if order == 1:
            ret += f"""
            1. '{self.trg_correct.sen}'
            2. '{self.trg_incorrect.sen}' 
            """
        else:
            ret += f"""
            1. '{self.trg_incorrect.sen}'
            2. '{self.trg_correct.sen}' 
            """

        return (order, ret)
    
    def to_affirming_prompts(self, shuffle=True) -> tuple[int, str, str]:
        """
        Returns two formatted prompts for likelihood evaulation
        """
        base = f"""
        This is a translation example of a text from English to French
        The context of the source sentence is '{self.src.pre}'
        The source sentence is '{self.src.sen}'
        The context translated in French is '{self.trg_correct.pre}'
        The correct translation is """
        
        order = random.randint(1, 2) if shuffle else 1
        if order == 1:
            return (order, base+f"'{self.trg_correct.sen}'", base+f"'{self.trg_incorrect.sen}'")
        if order == 2:
            return (order, base+f"'{self.trg_incorrect.sen}'", base+f"'{self.trg_correct.sen}'")
        
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
    def comparison_from_json(self, path: str) -> ComparisonTestSet:
        ...

class DiscourseMTLoader(TestSetLoader):
    SRC_TAG = "src"
    TRG_TAG = "trg"
    CORRECT_TAG = "correct"
    SEMI_CORRECT_TAG = "semi-correct"
    INCORRECT_TAG = "incorrect"

    def comparison_from_json(self, path: str) -> ComparisonTestSet:
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
