import json
import os
import re
import tiktoken

from datetime import datetime
from openai import OpenAI
from pprint import pprint
from sklearn.metrics import accuracy_score
from tacos.data import *
from tacos.prompt import *
from tacos.model import *
from tacos.utils import *
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

def openai_send():
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


def openai_parse():
    client = OpenAI()

    prompts_file = "f.json"
    with open(prompts_file, "r") as f:
        prompts = f.read()

    output_file = "file-"
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

def comparison_eval(testset: ComparisonTestSet, fmt: PromptFormatter, model: Model, save_outputs=None, max_tokens: None | int=None):
    trues = []
    preds = []
    
    for pair in tqdm(testset.pairs):
        res = fmt.format(pair)
        prompt = res["prompts"][0]
        trues.append(res["true"])
        output = model.get_completion(prompt=prompt, max_tokens=max_tokens)["choices"][0]["message"]["content"]
        pred = 0
        try:
            matched = re.search(r"choice:\s*(\d+)", output)
            pred = 0 if matched is None else int(matched.group(1))
        except ValueError:
            pass

        preds.append(pred)
        
        if save_outputs is not None:
            content = {
                "prompt": prompt,
                "output": output,
                "true": res["true"],
                "pred": pred
            }
            append_to_file(save_outputs, json.dumps(content) + "\n")

    return trues, preds

def comparison_eval_json(testset: ComparisonTestSet, fmt: PromptFormatter, model: Model, save_outputs=None):
    trues = []
    preds = []
    
    for pair in tqdm(testset.pairs):
        res = fmt.format(pair)
        prompt = res["prompts"][0]
        schema = res["schema"]

        output = model.get_completion_json(prompt, schema)["choices"][0]["message"]["content"]
        
        pred = 0
        try:
            pred = json.loads(output)["choice"]
        except:
            # Fallback for incorrect json
            matched = re.search(r"choice:\s*(\d+)", output)
            pred = 0 if matched is None else int(matched.group(1))

        trues.append(res["true"])
        preds.append(pred)
        
        if save_outputs is not None:
            content = {
                "id": pair.metadata["id"],
                "type": pair.metadata["type"],
                "prompt": prompt,
                "output": output,
                "true": res["true"],
                "pred": pred
            }
            append_to_file(save_outputs, json.dumps(content) + "\n")

    return trues, preds

def likelihood_eval(testset: ComparisonTestSet, fmt: PromptFormatter, model, tokenizer):
    trues = []
    preds = []
    for example in tqdm(testset.pairs):
        res = fmt.format_user(example=example)
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

def comparison_experiments(save_outputs: bool=True):
    models: list[Model] = [
        LlamaCPPModel("Mistral7BInstruct", "./models/mistral/mistral-7B-Instruct-Q4_K_M.gguf"),
        LlamaCPPModel("Mistral8x7BInstruct", "./models/mistral/mistral-8x7B-Instruct-Q4_K_M.gguf"),
        LlamaCPPModel("Llama38BInstruct", "./models/llama3/llama3-8B-Q4_K_M.gguf"),
        LlamaCPPModel("Llama370BInstruct", "./models/llama3/llama3-70B-Q4_K_M.gguf")
    ]

    loader = DiscourseMTLoader()
    testsets = {
        "lexical-choice": loader.comparison_from_lexical_choice("./discourse-mt-test-sets/test-sets/lexical-choice.json"),
        "anaphora": loader.comparison_from_anaphora("./discourse-mt-test-sets/test-sets/anaphora.json"),
    }
    
    do_cross_params = True
    all_params = {
        "anaphora": [
            {
                "translate_context": False,
                "explanation_instructions": None
            },
            {
                "translate_context": True,
                "explanation_instructions": None
            }
        ]
    }

    all_params_cross = {
            "anaphora": {
                "translate_context": [True],
                "explanation_instructions": [
                    None,
                    "For explanation purposes, what does the pronoun refer to in the sentence you chose?",
                ]
        },
            "lexical-choice": {
                "translate_context": [True],
                "explanation_instructions": [
                    None,
                    "For explanation purposes, write a short definition of each of the two words",
                    "For explanation purposes, why did you not choose the other word?",
                    "For explanation purposes, write an example sentence for each one of the two words.",
                ]
        }
    }

    if save_outputs:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        results_filename = f"""results-{timestamp}.json"""

        try:
            os.mkdir(timestamp)
        except:
            print("Could not create results dir")
            exit(1)

    results = []
    for model in models:
        model.load(n_threads=16)
        for testsetname, testset in testsets.items():
            params = params_product(**all_params_cross[testsetname]) if do_cross_params else all_params[testsetname]
            for i, p in enumerate(params):
                fmt = ENFRChoiceFormatterJSON(translate_context=p["translate_context"], explanation_instructions=p["explanation_instructions"])

                print(f"MODEL: {model.name}")
                print(f"TESTSET: {testsetname}")
                print(f"FMT: {fmt.__class__.__name__}")
                print(f"PARAMS:")
                pprint(p)

                outputs_filename = None
                if save_outputs:
                    metadata_filename = os.path.join(timestamp, f"""outputs-{model.name}-{testsetname}-{timestamp}-metadata.jsonl""")
                    outputs_filename = os.path.join(timestamp, f"""outputs-{model.name}-{testsetname}-{timestamp}-{i}.jsonl""")
                    
                    append_to_file(metadata_filename, json.dumps({"id": i, "params": p}) + "\n")
                
                trues, preds = comparison_eval_json(testset, fmt, model, save_outputs=outputs_filename)

                result = {
                    "model": model.name,
                    "testset": testsetname,
                    "params": p,
                    #"accuracy": accuracy_score(trues, preds),
                    "failed": sum(map(lambda p: 1 if p not in [1, 2] else 0, preds))
                }

                results.append(result)
        model.unload()
            
    pprint(results)
    with open(results_filename, "w") as f:
        f.write(json.dumps(results))

def main():
    comparison_experiments()

if __name__ == "__main__":
    main()
