"""TTFT / ITL bench against a running vLLM server.
Fixed prompt set (frozen), 3 warmup + 30 measured requests, report p50/p95.
Run single-stream: this measures the format, not the scheduler."""
# TODO: openai-client loop against localhost:8000, timestamp first token vs rest.
