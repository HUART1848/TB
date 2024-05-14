from sklearn.metrics import accuracy_score, recall_score, f1_score

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from tqdm import tqdm

from pprint import pprint
from tacos.data import DiscourseMTLoader

def simple_comparison_mistral():
    path = "discourse-mt-test-sets/test-sets/anaphora.json"
    testset = DiscourseMTLoader().comparison_from_json(path=path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    trues = []
    preds = []
    for example in tqdm(testset.examples):
        true, prompt = example.to_choice_prompt(shuffle=False)
        trues.append(true)

        message = [{"role": "user", "content" : prompt}]
        inputs = tokenizer.apply_chat_template(message, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=2, eos_token_id=2)

        pred = tokenizer.batch_decode(outputs[0, inputs.shape[1]:].unsqueeze(0))
        try:
            preds.append(int("".join(filter(str.isdigit, pred))))
        except ValueError:
            preds.append(0)

    print(f"accuracy: {accuracy_score(trues, preds)}")
    print(f"missing: {sum(filter(lambda x: x == 0, preds))}")

def simple_likelihood_mistral():
    path = "discourse-mt-test-sets/test-sets/anaphora.json"
    testset = DiscourseMTLoader().comparison_from_json(path=path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    trues = []
    preds = []
    for example in tqdm(testset.examples):
        true, a, b = example.to_affirming_prompts(shuffle=False)
        trues.append(true)

        inputs_a = tokenizer.encode(a, add_special_tokens=False, return_tensors="pt").to("cuda")
        outputs_a = model(inputs_a, labels=inputs_a)
        log_p_a = -outputs_a.loss * inputs_a.shape[1]

        inputs_b = tokenizer.encode(b, add_special_tokens=False, return_tensors="pt").to("cuda")
        outputs_b = model(inputs_b, labels=inputs_b)
        log_p_b = -outputs_b.loss * inputs_b.shape[1]

        pred = 1 if log_p_a > log_p_b else 2
        preds.append(pred)

    print(f"accuracy: {accuracy_score(trues, preds)}")

def simple_likelihood_nllb():
    path = "discourse-mt-test-sets/test-sets/anaphora.json"
    testset = DiscourseMTLoader().comparison_from_json(path=path)

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    trues = []
    preds = []
    for example in tqdm(testset.examples):
        true, a, b = example.to_affirming_prompts(shuffle=False)
        trues.append(true)

        inputs_a = tokenizer.encode(a, add_special_tokens=False, return_tensors="pt")
        outputs_a = model(inputs_a, labels=inputs_a)
        log_p_a = -outputs_a.loss * inputs_a.shape[1]

        inputs_b = tokenizer.encode(b, add_special_tokens=False, return_tensors="pt")
        outputs_b = model(inputs_b, labels=inputs_b)
        log_p_b = -outputs_b.loss * inputs_b.shape[1]

        pred = 1 if log_p_a > log_p_b else 2
        preds.append(pred)

    print(f"accuracy: {accuracy_score(trues, preds)}")

def main():
    #simple_comparison_mistral()
    #simple_likelihood_mistral()
    simple_likelihood_nllb()

if __name__ == "__main__":
    main()
