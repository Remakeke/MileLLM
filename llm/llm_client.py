import httpx
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             http_client=httpx.Client())

    def call(self, prompt, max_tokens=800, temperature=0.8):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=1.0
        )
        return response.choices[0].message.content
