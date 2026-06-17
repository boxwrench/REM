"""Probe to confirm the NPU embeddings endpoint capability."""

import json
from pathlib import Path
from rem.npu_client import NpuClient
from rem.config import Settings


def probe_embeddings() -> dict:
    """Probes the NPU embeddings endpoint and writes results to bench/e0_embeddings.json."""
    settings = Settings()
    client = NpuClient(settings)
    
    result = {
        "endpoint": f"http://localhost:{settings.npu_server_port}/v1/embeddings",
        "supported": False,
        "detail": "",
    }
    
    try:
        # We try to use the summarizer_model since it's running on the server
        model_name = settings.embedding_model or settings.summarizer_model or "llama3.2:1b"
        embeddings = client.embed(["Hello world"], model=model_name)
        
        if embeddings and len(embeddings) > 0 and isinstance(embeddings[0], list):
            result["supported"] = True
            result["detail"] = f"Successfully generated embedding: {len(embeddings[0])} dimensions."
            result["embedding_sample"] = embeddings[0][:5]
        else:
            result["detail"] = f"Returned empty or invalid format: {embeddings}"
    except Exception as e:
        result["detail"] = f"Failed to generate embeddings: {str(e)}"
        
    bench_dir = Path("bench")
    bench_dir.mkdir(exist_ok=True)
    with open(bench_dir / "e0_embeddings.json", "w") as f:
        json.dump(result, f, indent=2)
        
    return result


if __name__ == "__main__":
    res = probe_embeddings()
    print(json.dumps(res, indent=2))
