import gc

from llama_cpp import Llama

class Model:
    def load(self):
        ...

    def unload(self):
        ...

    def get_completion(self, prompt: str, max_tokens: int=64) -> str:
        ...
    

class MistralInstructModel(Model):
    MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
    
    model = None

    def load(self, verbose: bool=False):
        self.model = Llama("./models/mistral/mistral-7B-Q4_K_M.gguf", verbose=verbose)

    def unload(self):
        del self.model
        gc.collect()

    def get_completion(self, prompt: str, max_tokens: int=64) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        return(self.model(prompt, max_tokens=10))

class Llama3InstructModel(Model):
    MODEL_PATH = "./models/llama3/llama3-8B-Instruct-Q4_K_M.gguf"
    
    model = None

    def load(self, verbose: bool=False):
        self.model = Llama(self.MODEL_PATH, verbose=verbose)

    def unload(self):
        del self.model
        gc.collect()

    def get_completion(self, prompt: str, max_tokens: int=64) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        return(self.model(prompt, max_tokens=max_tokens))
