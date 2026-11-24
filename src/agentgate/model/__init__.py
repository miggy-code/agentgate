"""What the gate needs to decide a model call: the request's facts (`admission`), its worst
case cost (`pricing`), the run tokens a proxy accepts (`tokens`) and the provider keys it
holds (`keys`). The HTTP proxy that uses them is `agentgate.model_proxy`."""
