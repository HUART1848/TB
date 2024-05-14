from tacos.data import ComparisonExample

class PromptFormatter():
    def format(self) -> str:
        ...

class SimpleENFRPromptFormatter(PromptFormatter):
    def format(self, example: ComparisonExample) -> str:
        return f"""
        You will decide wich of two translations from english to french is the most correct one.
        The context of the original sentence is '{example.src.pre}'
        The original sentence is '{example.src.sen}'
        The context translated in french is '{example.trg_correct.pre}'
        Which of the following is correct? Answer only with '1' or '2'
        1. '{example.trg_correct.sen}'
        2. '{example.trg_incorrect.sen}' 
        """
        
