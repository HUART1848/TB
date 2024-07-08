import gc
import json
import tiktoken
import torch

from datetime import datetime
from openai import OpenAI
from pprint import pprint
from sklearn.metrics import accuracy_score, confusion_matrix
from tacos.data import *
from tacos.prompt import *
from tacos.model import *
from tqdm import tqdm

def get_completion(
    client,
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs
    }

    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def f():
    path = "discourse-mt-test-sets/test-sets/lexical-choice.json"
    testset = DiscourseMTLoader().comparison_from_lexical_choice(path=path)

    print(testset)

    client = OpenAI()
    encoding = tiktoken.get_encoding("cl100k_base")

    fmt = OpenAIBatchPromptFormatter(ENFRChoiceFormatter(), model="gpt-4o", max_tokens=1)
    prompts, batch = fmt.to_json_batch(testset)

    now = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    batchname = f"batch_{now}.jsonl"
    with open(batchname, "w") as f:
        f.write(batch)

    with open(f"prompts-{now}.json", "w") as f:
        f.write(json.dumps(prompts))

    batch_input_file = client.files.create(
        file=open(batchname, "rb"),
        purpose="batch"
    )

    ret = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "comparison eval"
        }
    )


def f2():
    client = OpenAI()

    prompts_file = "prompts-2024-06-21T130448.json"
    with open(prompts_file, "r") as f:
        prompts = f.read()

    output_file = "file-K3OLlGOwgZMRQR5NQsUSlioB"
    results = client.files.content(output_file)

    prompts = json.loads(prompts)
    trues = list()
    preds = list()
    for i, l in enumerate(results.iter_lines()):
        true = prompts[i]["true"]

        try:
            pred = int(json.loads(l)["response"]["body"]["choices"][0]["message"]["content"])
        except ValueError:
            pred = 0

        trues.append(true)
        preds.append(pred)

    print(accuracy_score(trues, preds))

def comparison_eval(testset: ComparisonTestSet, fmt: PromptFormatter, model: Model):
    trues = []
    preds = []
    for example in tqdm(testset.examples):
        res = fmt.format(example)
        prompt = res["prompts"][0]
        trues.append(res["true"])

        pred = model.get_completion(prompt=prompt, max_tokens=2)

        try:
            preds.append(int("".join(filter(str.isdigit, pred))))
        except ValueError:
            preds.append(0)

    return trues, preds


def likelihood_eval(testset: ComparisonTestSet, fmt: PromptFormatter, model, tokenizer):
    trues = []
    preds = []
    for example in tqdm(testset.examples):
        res = fmt.format(example=example)
        a, b = res["prompts"][0], res["prompts"][1]
        trues.append(res["true"])

        inputs_a = tokenizer.encode(a, add_special_tokens=False, return_tensors="pt")
        outputs_a = model(inputs_a, labels=inputs_a)
        log_p_a = -outputs_a.loss * inputs_a.shape[1]
        
        inputs_b = tokenizer.encode(b, add_special_tokens=False, return_tensors="pt")
        outputs_b = model(inputs_b, labels=inputs_b)
        log_p_b = -outputs_b.loss * inputs_b.shape[1]

        pred = 1 if log_p_a > log_p_b else 2
        preds.append(pred)

    return accuracy_score(trues, preds)

def comparison_experiments():
    models = [
        Llama3InstructModel(),
        MistralInstructModel()
    ]

    for m in models:
        m.load()

    loader = DiscourseMTLoader()
    testsets = {
        "anaphora": loader.comparison_from_anaphora("./discourse-mt-test-sets/test-sets/anaphora.json"),
        "lexical-choice": loader.comparison_from_lexical_choice("./discourse-mt-test-sets/test-sets/lexical-choice.json")
    }
    
    params = {
        "translate_context": [False, True]
    }

    results = []
    for model in models:
        for testsetname, testset in testsets.items():
            for p in params["translate_context"]:
                print(f"MODEL: {model.__class__.__name__}")

                fmt = ENFRChoiceFormatter(translate_context=True)
                print(f"FMT: {fmt.__class__.__name__}")
                trues, preds = comparison_eval(testset, fmt, model)
                print(confusion_matrix(trues, preds))

                for i in range(0, 3):
                    print(f"{i} : {sum([pred for pred in preds if pred == i])}")

                for i in range(0, 3):
                    print(f"{i} : {sum([tr for tr in trues if tr == i])}")

                result = {
                    "model": model.__class__.__name__,
                    "testset": testsetname,
                    "translate_context": p,
                    "accuracy": accuracy_score(trues, preds)
                }

                pprint(result)
                results.append(result)
            break
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()
    
    pprint(results)

def likelihood_experiments():
    pass

def main():
    comparison_experiments()

if __name__ == "__main__":
    main()
