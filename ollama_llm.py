# ollama_llm.py

from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
import requests
import json

class Ollama(LLM):
    model_name: str = "llama2"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        }

        # Send the request with streaming enabled
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        generated_text = ''
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                try:
                    data = json.loads(line)
                    chunk = data.get('response', '')
                    generated_text += chunk

                    # Handle stop tokens
                    if stop and any(token in chunk for token in stop):
                        # Truncate at the first occurrence of any stop token
                        for token in stop:
                            if token in generated_text:
                                generated_text = generated_text.split(token)[0]
                                break
                        break
                except json.JSONDecodeError as e:
                    # If a line isn't valid JSON, we can skip it or handle it as needed
                    continue

        return generated_text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @property
    def _llm_type(self) -> str:
        return "ollama"
