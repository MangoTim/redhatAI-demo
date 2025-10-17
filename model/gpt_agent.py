from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class GPTAgent:
    def __init__(self):
        self.model_name = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.generator = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32),
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def __call__(self, prompt, **kwargs):
        print(f"Using model: {self.model_name}", flush=True)

        kwargs.setdefault("max_new_tokens", 256)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_k", 0)
        kwargs.setdefault("top_p", 1.0)

        outputs = self.generator(prompt, **kwargs)
        return [{"generated_text": outputs[0]["generated_text"]}]
