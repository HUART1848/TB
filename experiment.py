from sklearn.metrics import accuracy_score, recall_score, f1_score

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        true, prompt = example.to_prompt(shuffle=False)
        trues.append(true)

        message = [{"role": "user", "content" : prompt}]
        inputs = tokenizer.apply_chat_template(message, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=2)

        pred = tokenizer.batch_decode(outputs[0, inputs.shape[1]:].unsqueeze(0))
        try:
            preds.append(int("".join(filter(str.isdigit, pred))))
        except ValueError:
            preds.append(0)

    
    print(f"accuracy: {accuracy_score(trues, preds)}")
    print(f"recall: {recall_score(trues, preds, average='micro')}")
    print(f"f1-score: {f1_score(trues, preds, average='micro')}")

def main():
    simple_comparison_mistral()    

if __name__ == "__main__":
    main()
