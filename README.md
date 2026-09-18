# GridWise LLM

BUP CSE Fest 2026 Preliminary submission. Stage 1 scaffold: health endpoint only.
Full documentation (LLM provider, guardrails, optimizer, curl examples) lands before submission.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```
