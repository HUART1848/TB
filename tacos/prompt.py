import json
import random
import textwrap

from tacos.data import ComparisonPair, ComparisonTestSet

class PromptFormatter():
    def format(self, example: ComparisonPair) -> dict:
        """
        Returns a dict of shape {"true": ..., "prompts": [...]}.
        true is the 1-indexed position of the correct answer in the prompt(s)
        prompts may contain one or more prompts
        """
        ...

class ENFRChoiceFormatter(PromptFormatter):
    translate_context: bool = ...
    explanation_instructions: None|str = ...

    def __init__(self, translate_context: bool=False, explanation_instructions: None|str=None):
        self.translate_context = translate_context
        self.explanation_instructions = explanation_instructions

    def format(self, example: ComparisonPair, shuffle: bool=True) -> dict:
        """
        Returns a formatted prompt with the position of the correct sentence (1 or 2).
        """

        context = f"The context translated in French is '{example.trg_correct.pre}'" if self.translate_context else ""
        order = random.randint(1, 2) if shuffle else 1
        choices = ""
        if order == 1:
            choices += f"""
            1. '{example.trg_correct.sen}'
            2. '{example.trg_incorrect.sen}' 
            """
        else:
            choices += f"""
            1. '{example.trg_incorrect.sen}'
            2. '{example.trg_correct.sen}' 
            """

        prompt = f"""
        You will decide which of two translations from English to French is the most correct one.
        The context of the source sentence is '{example.src.pre}'
        The source sentence is '{example.src.sen}'
        {context}
        Which of the following translations is correct?
        {choices}
        {"" if self.explanation_instructions is None else self.explanation_instructions}
        MAKE SURE you only answer in the following manner:
        {"" if self.explanation_instructions is None else "explanation: (your explanation)"}
        choice: (1 or 2) END
        """

        prompt = textwrap.dedent(prompt)
        return {
            "true": order,
            "prompts" : [prompt]
        }
    
class ENFRChoiceFormatterJSON(PromptFormatter):
    translate_context: bool = ...
    explanation_instructions: None | str = ...

    def __init__(self, translate_context: bool=False, explanation_instructions:None|str=None):
        self.translate_context = translate_context
        self.explanation_instructions = explanation_instructions

    def format(self, example: ComparisonPair, shuffle: bool=True) -> dict:
        context = f"The context translated in French is '{example.trg_correct.pre}'" if self.translate_context else ""
        order = random.randint(1, 2) if shuffle else 1
        choices = ""
        if order == 1:
            choices = f"""
            1. '{example.trg_correct.sen}'
            2. '{example.trg_incorrect.sen}' 
            """
        else:
            choices = f"""
            1. '{example.trg_incorrect.sen}'
            2. '{example.trg_correct.sen}' 
            """

        prompt = f"""
        You will decide which of two translations from English to French is the most correct one.
        The context of the source sentence is '{example.src.pre}'
        The source sentence is '{example.src.sen}'
        {context}
        Which of the following translations is correct?
        {choices}
        {self.explanation_instructions if self.explanation_instructions is not None else ""}
        Answer as json with the field "choice": (1 or 2)"""

        schema = {}
        if self.explanation_instructions is None:
            schema = {
                "type": "object",
                "properties": {
                    "choice": {"type": "number"},
                },
                "required": ["choice"]
            }
        else:
            prompt += """ and the field "explanation" containing your explanation"""
            schema = {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "choice": {"type": "number"}
                },
                "required": ["explanation", "choice"]
            }

        prompt = textwrap.dedent(prompt)
        return {
            "true": order,
            "prompts": [prompt],
            "schema": schema
        }

class ENFRAffirmationFormatter(PromptFormatter):
    translate_context = True

    def __init__(self, translate_context: bool):
        self.translate_context = translate_context

    def format_user(self, example: ComparisonPair) -> dict:
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
            res = self.fmt.format_user(example)
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
