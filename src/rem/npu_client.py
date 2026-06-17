"""OpenAI-compatible client for Lemonade/FLM running on the Ryzen AI NPU."""

import httpx
from rem.config import Settings


class NpuUnavailable(Exception):
    """Raised when the NPU service is unreachable."""
    pass


class NpuClient:
    """Client for communicating with the NPU service."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.base_url = f"http://localhost:{self.settings.npu_server_port}"
        # Configure timeouts from settings
        self.timeout = httpx.Timeout(
            self.settings.npu_request_timeout_s,
            connect=self.settings.npu_connect_timeout_s,
        )


    def _post_with_retry(self, path: str, json_data: dict) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=json_data)
                response.raise_for_status()
                return response
        except (httpx.RequestError, httpx.HTTPStatusError):
            # Bounded retry - try exactly once more
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=json_data)
                    response.raise_for_status()
                    return response
            except (httpx.RequestError, httpx.HTTPStatusError) as final_exc:
                raise NpuUnavailable(
                    f"NPU service request failed at {url} after retry: {final_exc}"
                ) from final_exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        **extra_sampling: object,
    ) -> str:
        """Sends a chat completions request to the NPU endpoint.

        Args:
            messages: A list of message dictionaries.
            model: Optional model name override.
            max_tokens: Optional max tokens cap.
            temperature: Sampling temperature (default 0.0).
            **extra_sampling: Additional sampling parameters forwarded verbatim
                to the payload (e.g. frequency_penalty, repetition_penalty).
                FLM honors these; any it ignores are silently dropped by the server.

        Returns:
            The text response from the model.
        """
        payload = {
            "model": model or self.settings.summarizer_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra_sampling)  # frequency_penalty, repetition_penalty, etc.

        response = self._post_with_retry("/v1/chat/completions", payload)
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"Malformed OpenAI API response: {exc}") from exc

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Sends an embeddings request to the NPU endpoint.

        Args:
            texts: List of strings to embed.
            model: Optional model name override.

        Returns:
            A list of float lists representing the embeddings.
        """
        payload = {
            "model": model or self.settings.embedding_model,
            "input": texts,
        }
        response = self._post_with_retry("/v1/embeddings", payload)
        try:
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"Malformed OpenAI API response: {exc}") from exc

    def health(self) -> bool:
        """Checks the health of the NPU service using the v1/models endpoint."""
        url = f"{self.base_url}/v1/models"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                return response.status_code == 200
        except httpx.RequestError:
            # Retry once
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url)
                    return response.status_code == 200
            except httpx.RequestError:
                return False
