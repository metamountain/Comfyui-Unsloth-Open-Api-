"""
Unsloth API PRO - ComfyUI Custom Nodes
Two nodes: UnslothModelPicker + UnslothProAPI
Docs: https://unsloth.ai/docs/basics/api

Defaults tuned for Gemma 4 2B (gemma-4-2b-it / gemma-4-2b-it-GGUF)
"""

import numpy as np
import base64
import time
import requests
from io import BytesIO
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key.strip():
        h["Authorization"] = f"Bearer {api_key.strip()}"
    return h


def _fetch_model_list(base_url: str, api_key: str, timeout: int = 8) -> list:
    """GET /v1/models — official Unsloth OpenAI-compat endpoint."""
    try:
        r = requests.get(
            f"{base_url}/v1/models",
            headers=_build_headers(api_key),
            timeout=timeout,
        )
        r.raise_for_status()
        data   = r.json().get("data", [])
        models = [m["id"] for m in data if m.get("id")]
        return models if models else ["auto"]
    except Exception:
        return ["auto"]


def _get_active_model(base_url: str, api_key: str, timeout_s: int):
    """GET /api/inference/status → (model_id, gguf_variant, error_or_None)"""
    try:
        r = requests.get(
            f"{base_url}/api/inference/status",
            headers=_build_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data         = r.json()
        model_id     = (data.get("model_path") or data.get("model")
                        or data.get("model_name") or "unknown")
        gguf_variant = data.get("gguf_variant")
        if not data.get("loaded", True):
            return None, None, "No model loaded in Unsloth Studio — load one first."
        return model_id, gguf_variant, None
    except requests.exceptions.ConnectionError:
        return None, None, f"Cannot reach {base_url} — is Unsloth Studio running?"
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        if code == 401:
            return None, None, "401 Unauthorized — check your API key."
        return None, None, f"HTTP {code} from /api/inference/status"
    except Exception as exc:
        return None, None, f"Status query failed: {exc}"


# Module-level model list cache
_model_cache    = ["auto"]
_cache_base_url = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Node 1 — Unsloth Model Picker
# ─────────────────────────────────────────────────────────────────────────────

class UnslothModelPicker:
    """
    Queries GET /v1/models and exposes a live dropdown of every model
    loaded in Unsloth Studio.  Wire model_id → UnslothProAPI.model_id.
    Hit ComfyUI > Refresh to update the list after loading a new model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _fetch_model_list(_cache_base_url or "http://localhost:8888", "")
        if models and models != ["auto"]:
            global _model_cache
            _model_cache = models

        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "http://localhost:8888",
                    "tooltip": "Unsloth Studio base URL",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your sk-unsloth-... API key (leave blank if none set)",
                }),
                "model_id": (_model_cache, {
                    "tooltip": "Live list from GET /v1/models — Refresh ComfyUI to update",
                }),
            }
        }

    RETURN_TYPES    = ("STRING", "STRING", "STRING")
    RETURN_NAMES    = ("model_id", "api_url", "api_key")
    OUTPUT_TOOLTIPS = (
        "Model ID — wire into UnslothProAPI",
        "api_url  — wire into UnslothProAPI",
        "api_key  — wire into UnslothProAPI",
    )
    FUNCTION = "pick"
    CATEGORY = "Unsloth"

    @classmethod
    def IS_CHANGED(cls, api_url, api_key, model_id):
        global _model_cache, _cache_base_url
        base_url = api_url.rstrip("/")
        fresh    = _fetch_model_list(base_url, api_key)
        if fresh and fresh != ["auto"]:
            _model_cache    = fresh
            _cache_base_url = base_url
        return str(_model_cache)

    def pick(self, api_url, api_key, model_id):
        return (model_id, api_url.rstrip("/"), api_key)


# ─────────────────────────────────────────────────────────────────────────────
#  Node 2 — Unsloth API PRO
# ─────────────────────────────────────────────────────────────────────────────

# ── Gemma 4 2B recommended defaults (gemma-4-2b-it / GGUF) ────────────────
# Google recommends: temp=1.0, top_p=0.95, top_k=64 for Gemma 4 family.
# repetition_penalty min is 0.0 — llama.cpp accepts 0.0–2.0:
#   1.0 = neutral  |  >1.0 = discourage repeats  |  <1.0 = encourage repeats
# ──────────────────────────────────────────────────────────────────────────

_GEMMA4_TEMP        = 1.0
_GEMMA4_TOP_P       = 0.95
_GEMMA4_TOP_K       = 64
_GEMMA4_MIN_P       = 0.0
_GEMMA4_REP_PEN     = 1.1   # slight penalty keeps 2B output clean
_GEMMA4_PRES_PEN    = 0.0
_GEMMA4_MAX_TOKENS  = 2048


class UnslothProAPI:
    """
    Unsloth Studio inference node — Vision and Text modes.

    Connect UnslothModelPicker → model_id for a live model dropdown.
    Leave model_id disconnected (or set to 'auto') to use whatever
    is currently loaded in Studio (/api/inference/status).

    Outputs
    -------
    response      The model's reply
    thinking      Chain-of-thought (only when enable_thinking=True)
    active_model  Model ID that was actually used
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                # ┌─ CONNECTION ──────────────────────────────────────────────┐
                "api_url": ("STRING", {
                    "default": "http://localhost:8888",
                    "tooltip": "Unsloth Studio base URL — or wire from Model Picker",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "sk-unsloth-... API key — or wire from Model Picker",
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ MODE ────────────────────────────────────────────────────┐
                "mode": (["vision", "text"], {
                    "default": "vision",
                    "tooltip": "vision = image + text prompt  |  text = text only",
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ SYSTEM PROMPT ───────────────────────────────────────────┐
                # Static label rendered as a locked single-option dropdown
                "_header_system": (["🔧  SYSTEM PROMPT"],),
                "system_prompt": ("STRING", {
                    "multiline":      True,
                    "default":        "You are a helpful vision assistant. "
                                      "Analyse the provided image thoroughly "
                                      "and answer the user's question precisely.",
                    "tooltip":        "Defines the model's role and behaviour",
                    "dynamicPrompts": False,
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ USER PROMPT ─────────────────────────────────────────────┐
                # Static label rendered as a locked single-option dropdown
                "_header_user": (["💬  USER PROMPT"],),
                "user_prompt": ("STRING", {
                    "multiline":      True,
                    "default":        "Describe this image in detail.",
                    "tooltip":        "Your question or instruction to the model",
                    "dynamicPrompts": False,
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ SAMPLING — Gemma 4 2B defaults ─────────────────────────┐
                "temperature": ("FLOAT", {
                    "default": _GEMMA4_TEMP,
                    "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Gemma 4 recommended: 1.0  |  0 = deterministic",
                }),
                "top_p": ("FLOAT", {
                    "default": _GEMMA4_TOP_P,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Gemma 4 recommended: 0.95",
                }),
                "top_k": ("INT", {
                    "default": _GEMMA4_TOP_K,
                    "min": -1, "max": 200, "step": 1,
                    "tooltip": "Gemma 4 recommended: 64  |  -1 = disabled",
                }),
                "min_p": ("FLOAT", {
                    "default": _GEMMA4_MIN_P,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Min probability filter — 0.0 = disabled",
                }),
                # FIX: min=0.0 (not 1.0) — llama.cpp accepts 0.0–2.0
                #      1.0 = neutral  |  >1.0 = reduce repeats
                "repetition_penalty": ("FLOAT", {
                    "default": _GEMMA4_REP_PEN,
                    "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "1.0 = no effect  |  >1.0 = reduce repetition  "
                               "(min is 0.0, NOT 1.0 — fixed)",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": _GEMMA4_PRES_PEN,
                    "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalise tokens that appeared earlier in the text",
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ OUTPUT ──────────────────────────────────────────────────┐
                "max_tokens": ("INT", {
                    "default": _GEMMA4_MAX_TOKENS,
                    "min": 1, "max": 32768, "step": 64,
                    "tooltip": "Gemma 4 2B context window: 8 192 tokens",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1, "max": 2**32 - 1, "step": 1,
                    "tooltip": "-1 = random (server default)",
                }),
                "timeout_s": ("INT", {
                    "default": 120,
                    "min": 30, "max": 600, "step": 10,
                    "tooltip": "Request timeout in seconds",
                }),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0, "max": 5, "step": 1,
                    "tooltip": "Auto-retry on timeout / connection drops (exponential back-off)",
                }),
                # └──────────────────────────────────────────────────────────┘

                # ┌─ UNSLOTH EXTRAS ──────────────────────────────────────────┐
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Emit chain-of-thought in the 'thinking' output socket",
                }),
                "reasoning_effort": (["none", "low", "medium", "high"], {
                    "default": "none",
                    "tooltip": "none = disable  |  high = deepest reasoning (slower)",
                }),
                # └──────────────────────────────────────────────────────────┘
            },

            "optional": {
                # Image only needed in vision mode
                "image": ("IMAGE",),
                # Wire from UnslothModelPicker, or type a model ID, or leave 'auto'
                "model_id": ("STRING", {
                    "default": "auto",
                    "tooltip": "Wire from Model Picker for a live dropdown. "
                               "'auto' = use whatever is currently loaded in Studio.",
                }),
            },
        }

    RETURN_TYPES    = ("STRING", "STRING", "STRING")
    RETURN_NAMES    = ("response", "thinking", "active_model")
    OUTPUT_TOOLTIPS = (
        "The model's reply text",
        "Chain-of-thought / reasoning (only when enable_thinking=True)",
        "Model ID that was actually used for this inference",
    )
    FUNCTION  = "chat"
    CATEGORY  = "Unsloth"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tensor_to_b64_jpeg(image_tensor) -> str:
        arr = np.clip(255.0 * image_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Main ──────────────────────────────────────────────────────────────────

    def chat(
        self,
        api_url, api_key, mode,
        _header_system, system_prompt,
        _header_user, user_prompt,
        temperature, top_p, top_k, min_p,
        repetition_penalty, presence_penalty,
        max_tokens, seed, timeout_s, max_retries,
        enable_thinking, reasoning_effort,
        image=None, model_id="auto",
    ):
        base_url  = api_url.rstrip("/")
        is_vision = (mode == "vision")

        # 1 ── Resolve model ──────────────────────────────────────────────────
        gguf_variant = None
        if not model_id or model_id.strip().lower() in ("auto", ""):
            resolved, gguf_variant, err = _get_active_model(base_url, api_key, timeout_s)
            if err:
                return (f"ERROR: {err}", "", "none")
            model_id    = resolved
            model_label = f"{model_id} [{gguf_variant}]" if gguf_variant else model_id
        else:
            model_label = model_id

        # 2 ── Vision guard ───────────────────────────────────────────────────
        if is_vision and image is None:
            return (
                "ERROR: mode='vision' but no IMAGE is connected.\n"
                "Connect an image node or switch mode to 'text'.",
                "",
                model_label,
            )

        # 3 ── Build user content ─────────────────────────────────────────────
        if is_vision:
            img_b64 = self._tensor_to_b64_jpeg(image)
            user_content = [
                {"type": "text",      "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                }},
            ]
        else:
            img_b64      = None
            user_content = user_prompt

        # 4 ── Payload ────────────────────────────────────────────────────────
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "temperature":        temperature,
            "top_p":              top_p,
            "top_k":              top_k,
            "min_p":              min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty":   presence_penalty,
            "max_tokens":         max_tokens,
            "stream":             False,
            "enable_thinking":    enable_thinking,
        }

        if img_b64:                    payload["image_base64"]     = img_b64
        if gguf_variant:               payload["gguf_variant"]     = gguf_variant
        if reasoning_effort != "none": payload["reasoning_effort"] = reasoning_effort
        if seed != -1:                 payload["seed"]             = seed

        # 5 ── Send with exponential back-off retry ───────────────────────────
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=_build_headers(api_key),
                    timeout=timeout_s,
                )
                r.raise_for_status()
                msg      = r.json()["choices"][0]["message"]
                content  = msg.get("content") or ""
                thinking = msg.get("reasoning_content") or msg.get("thinking") or ""
                return (content, thinking, model_label)

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                last_err = str(exc)
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[UnslothProAPI] Retry {attempt + 1}/{max_retries} in {wait}s …")
                    time.sleep(wait)
                    continue
                return (
                    f"ERROR: Connection failed after {max_retries + 1} attempts — {last_err}",
                    "", model_label,
                )

            except requests.exceptions.HTTPError:
                code = r.status_code
                detail = {
                    401: "401 Unauthorized — create / check your API key in "
                         "Unsloth Studio → Settings → API.",
                    404: "404 — no model loaded in Unsloth Studio yet. "
                         "Open Studio, load a model, and retry.",
                    422: f"422 Validation error — {r.text[:400]}",
                    500: f"500 Server error — {r.text[:200]}",
                }
                return (
                    f"ERROR: {detail.get(code, f'HTTP {code} — {r.text[:200]}')}",
                    "", model_label,
                )

            except KeyError:
                return (f"ERROR: Unexpected response format — {r.text[:200]}", "", model_label)
            except Exception as exc:
                return (f"ERROR: {exc}", "", model_label)

        return (f"ERROR: All retries exhausted — {last_err}", "", model_label)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "UnslothModelPicker": UnslothModelPicker,
    "UnslothProAPI":      UnslothProAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnslothModelPicker": "🔍 Unsloth Model Picker",
    "UnslothProAPI":      "✨ Unsloth API PRO",
}