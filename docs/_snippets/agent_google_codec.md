## Host Codec {#host-codec}

{{< meta agent_name >}} uses Gemini's `generateContent` wire to communicate with the Inspect bridge. The required `google-genai` codec installs automatically with `inspect-swe`.

The bridge selects its Google request handler from the wire format, not the model that actually serves the request. Therefore this applies even when Inspect routes an agent's requests to Anthropic, OpenAI, or a mock model. The codec does not require a Google account, API key, or sign-in.
