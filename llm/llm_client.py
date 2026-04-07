import httpx
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key, base_url):
        """
        Initialize OpenAI client with optional proxy support.
        """
        client_args = {}
        client_args["http_client"] = httpx.Client()

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_args
        )

    def call(self, prompt, max_tokens=800, temperature=0.8):
        """
        Send a prompt to the LLM and return the generated response.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated text response.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0
        )

        return response.choices[0].message.content