import gc

from llama_cpp import Llama

class Model:
    """Interface for generic LLMs"""
    name: str = ...

    def load(self): ...
    def unload(self): ...
    def get_completion(self, prompt: str) -> str: ...

class LlamaCPPModel(Model):
    """Frontend for `llama.cpp`-compatible models"""
    model_path: str = ...
    model: None | Llama = ...

    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path
        
    def load(self, verbose: bool=False, n_threads: int=8):
        self.model = Llama(model_path=self.model_path, verbose=verbose, n_threads=n_threads, n_threads_batch=n_threads)

    def unload(self):
        del self.model
        gc.collect()

    def get_completion(self, prompt: str, max_tokens: int=64, stop: list[str]=[]) -> str:
        if self.model is None:
            raise RuntimeError(f"Model {self.name} at {self.model_path} is not loaded")
        
        messages = [{"role": "user", "content": prompt}]
        return self.model.create_chat_completion(messages=messages, max_tokens=max_tokens, stop=stop)
