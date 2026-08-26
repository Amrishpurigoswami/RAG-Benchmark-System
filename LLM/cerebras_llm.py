import os

from openai import OpenAI


class CerebrasLLM:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1"
        )

        self.primary_model = os.getenv(
            "PRIMARY_MODEL"
        )

        self.fallback_model = os.getenv(
            "FALLBACK_MODEL"
        )

        self.second_fallback_model = os.getenv(
            "SECOND_FALLBACK_MODEL"
        )

    def _call_model(
        self,
        model,
        prompt
    ):

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    def generate(
        self,
        prompt
    ):

        models = [
            self.primary_model,
            self.fallback_model,
            self.second_fallback_model
        ]

        last_exception = None

        for index, model in enumerate(models):

            if not model:
                continue

            try:

                if index > 0:
                    print(
                        f"\nSwitching to fallback model: {model}"
                    )

                return self._call_model(
                    model,
                    prompt
                )

            except Exception as e:

                last_exception = e

                print(
                    f"\nModel '{model}' failed."
                )

                print(f"Reason: {e}")

        raise RuntimeError(
            f"All configured models failed.\n"
            f"Last error: {last_exception}"
        )