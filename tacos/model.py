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
    MODEL_PATH = "./models/mistral/mistral-7B-Q4_K_M.gguf"
    
    model: Llama = None


    def load(self, verbose: bool=False, n_threads=16):
        self.model = Llama(model_path=self.MODEL_PATH, verbose=verbose, n_threads=n_threads, n_threads_batch=n_threads)

    def unload(self):
        del self.model
        gc.collect()

    def get_completion(self, prompt: str, max_tokens: int=64) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        message = [{"role": "user", "content": prompt}]
        ret = self.model.create_chat_completion(messages=message, max_tokens=max_tokens, stop=["END"])
        return ret

class Llama3InstructModel(Model):
    MODEL_PATH = "./models/llama3/llama3-8B-Q4_K_M.gguf"
    
    model = None

    def load(self, verbose: bool=False, n_threads=16):
        self.model = Llama(model_path=self.MODEL_PATH, verbose=verbose, n_threads=n_threads, n_threads_batch=n_threads)

    def unload(self):
        del self.model
        gc.collect()

    def get_completion(self, prompt: str, max_tokens: int=64) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        message = [{"role": "user", "content": prompt}]
        ret = self.model.create_chat_completion(messages=message, max_tokens=max_tokens, stop=["END"])
        return ret
