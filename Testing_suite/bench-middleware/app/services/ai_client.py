import os
import httpx

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1/chat/completions")

SYSTEM_PROMPT = (
    "You are an expert in physics code analysis. Perform a Teleological Decomposition: "
    "Compare the provided code's implementation to its intended physics goal. "
    "Identify any architectural drift, security risks, or violations of the Axioms (Traversal, Sanitization, Parsimony). "
    "Return a concise markdown report."
)

async def analyze_code(filepath: str, code_snippet: str) -> str:
    payload = {
        "model": "local-llm",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"File: {filepath}\n\n{code_snippet}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    async with httpx.AsyncClient(base_url=LM_STUDIO_URL) as client:
        response = await client.post("", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # LM Studio returns OpenAI-compatible response
        return data["choices"][0]["message"]["content"]
