"""The model proxy: an OpenAI-compatible HTTP endpoint that puts every model request through
the gate. `proxy` serves one run (`agentgate model-proxy serve`); `server` serves many runs
started over a Unix control socket (`agentgate model-proxy server`), which `client` speaks."""
