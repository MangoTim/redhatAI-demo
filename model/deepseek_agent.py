import requests

class DeepSeekAgent:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.api_url = "http://modelserver:5000/v1/completions"

    def __call__(self, prompt, **kwargs):
        print(f"Using model: {self.model_name}", flush=True)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_new_tokens", 128),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 0),
            "do_sample": kwargs.get("do_sample", False)
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return [{"generated_text": result["choices"][0]["text"]}]
        else:
            return [{"generated_text": f"Error: {response.status_code} - {response.text}"}]