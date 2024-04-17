from llama_cpp import Llama

from pprint import pprint
from tacos.data import DiscourseMTLoader
from tacos.prompt import SimpleENFRPromptFormatter

def simple_comparison_mistral():
    path = "discourse-mt-test-sets/test-sets/anaphora.json"
    testset = DiscourseMTLoader().comparison_from_json(path=path)

    
    model_path = "models/mistral/7b/ggml-model-Q4_K_M.gguf"
    llm = Llama(model_path=model_path)

    prompt_fmt = SimpleENFRPromptFormatter()
    prompt = prompt_fmt.format(testset.examples[0])

    output = llm(
      prompt, # Prompt
      max_tokens=2, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    )

    print(output)

def main():
    simple_comparison_mistral()    

if __name__ == "__main__":
    main()
