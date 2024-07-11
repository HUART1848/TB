import json
import random
import textwrap

from tacos.data import ComparisonExample, ComparisonTestSet

class PromptFormatter():
    def format(self, example: ComparisonExample) -> dict:
        """
        Returns a dict of shape {"true": ..., "prompts": [...]}.
        true is the 1-indexed position of the correct answer in the prompt(s)
        prompts may contain one or more prompts
        """
        ...

class ENFRChoiceFormatter(PromptFormatter):
    translate_context = True

    def __init__(self, translate_context: bool):
        self.translate_context = translate_context

    def format(self, example: ComparisonExample, shuffle=True) -> dict:
        """
        Returns a formatted prompt with the position of the correct sentence (1 or 2).
        """
        context = f"The context translated in French is '{example.trg_correct.pre}'" if self.translate_context else ""
        ret = f"""
        You will decide which of two translations from English to French is the most correct one.
        The context of the source sentence is '{example.src.pre}'
        The source sentence is '{example.src.sen}'
        {context}
        Which of the following translations is correct? MAKE SURE you only answer with the following manner:
        START,choice=(1 or 2),END
        """

        order = random.randint(1, 2) if shuffle else 1
        if order == 1:
            ret += f"""
            1. '{example.trg_correct.sen}'
            2. '{example.trg_incorrect.sen}' 
            """
        else:
            ret += f"""
            1. '{example.trg_incorrect.sen}'
            2. '{example.trg_correct.sen}' 
            """

        ret = textwrap.dedent(ret)
        return {
            "true": order,
            "prompts" : [ret]
        }
    
class ENFRAffirmationFormatter(PromptFormatter):
    translate_context = True

    def __init__(self, translate_context: bool):
        self.translate_context = translate_context

    def format(self, example: ComparisonExample) -> dict:
        """
        Returns two formatted prompts for likelihood evaluation.
        The first prompt corresponds to the correct assessment. 
        """

        context = f"The context translated in French is '{example.trg_correct.pre}'" if self.translate_context else ""
        base = f"""
        This is a translation example of a text from English to French
        The context of the source sentence is '{example.src.pre}'
        The source sentence is '{example.src.sen}'
        {context}
        The correct translation is """
        
        a = textwrap.dedent(base + f"'{example.trg_correct.sen}'")
        b = textwrap.dedent(base + f"'{example.trg_incorrect.sen}'")

        return {
            "true": 1,
            "prompts": [a, b]
        }


class BatchPromptFormatter():
    fmt: PromptFormatter = ...

    def __init__(self, fmt: PromptFormatter):
        self.fmt = fmt

class OpenAIBatchPromptFormatter(BatchPromptFormatter):
    params = dict()
    
    def __init__(self, fmt: PromptFormatter, **kwargs):
        super().__init__(fmt)

        for (k, v) in kwargs.items():
            self.params[k] = v

    def to_json_batch(self, testset: ComparisonTestSet) -> tuple[list[dict], str]:
        """Returns each prompt and the resulting batch."""
        prompts = list()
        batch = ""
        for i, example in enumerate(testset.examples):
            res = self.fmt.format(example)
            prompts.append(res)

            for j, p in enumerate(res["prompts"]):
                cur = {
                    "custom_id": f"{i}-{j}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.params["model"],
                        "messages": [
                            {
                                "role": "user",
                                "content": p
                            }
                        ],
                        "max_tokens": self.params["max_tokens"]
                    }
                }
                batch += json.dumps(cur) + "\n"

        return (prompts, batch)
