"""Forwarder: convert Anthropic-style requests to OpenAI-style and proxy them.

Provides `start_forwarder_server(target_url, api_key, host, port)` which runs
a small Flask app exposing POST /v1/forward accepting Anthropic-like JSON and
forwarding to an OpenAI-compatible endpoint, returning an Anthropic-like response.
"""

from typing import Dict


def _convert_anthropic_to_openai_body(anthropic_json: dict) -> dict:
    body = {}
    if "model" in anthropic_json:
        body["model"] = anthropic_json["model"]
    else:
        body["model"] = anthropic_json.get("model_name", "gpt-4o")

    msgs = anthropic_json.get("messages") or anthropic_json.get("input") or []
    openai_msgs = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content") or m.get("text") or ""
        openai_msgs.append({"role": role, "content": content})

    body["messages"] = openai_msgs
    for k in ("temperature", "max_tokens", "top_p", "n", "stop"):
        if k in anthropic_json:
            body[k] = anthropic_json[k]

    return body


def _convert_openai_to_anthropic(openai_resp: dict) -> dict:
    anth = {
        "id": openai_resp.get("id"),
        "object": openai_resp.get("object"),
        "model": openai_resp.get("model"),
    }
    assistant_text = None
    choices = openai_resp.get("choices") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        assistant_text = msg.get("content") or first.get("text")
        anth["raw_choice"] = first

    anth["completion"] = {"role": "assistant", "content": assistant_text}
    return anth


def start_forwarder_server(target_url: str, api_key: str = None, host: str = "0.0.0.0", port: int = 7860):
    from flask import Flask, request, jsonify
    import requests

    app = Flask(__name__)

    @app.route("/v1/forward", methods=["POST"])
    def forward():
        try:
            anthropic_json = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "invalid json"}), 400

        openai_body = _convert_anthropic_to_openai_body(anthropic_json)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = requests.post(target_url, headers=headers, json=openai_body, timeout=60)
        except Exception as e:
            return jsonify({"error": "failed to reach target", "detail": str(e)}), 502

        try:
            openai_resp = resp.json()
        except Exception:
            return (resp.text, resp.status_code, {"Content-Type": "text/plain"})

        anth_resp = _convert_openai_to_anthropic(openai_resp)
        anth_resp["openai_response"] = openai_resp
        return jsonify(anth_resp), resp.status_code

    print(f"Starting forwarder -> target: {target_url} on {host}:{port}")
    app.run(host=host, port=port)


__all__ = ["start_forwarder_server"]
