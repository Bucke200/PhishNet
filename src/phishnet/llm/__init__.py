"""Phase 4 LLM layer (Groq `openai/gpt-oss-120b`, prompt `p4-v1`).

Registration pins the route; this package holds its frozen artifacts. No
network calls happen on import. The §4.4 gate runs before `client.py` lands;
`schema.py` + `prompts/p4-v1.txt` are the exact artifacts the gate tests.
"""
