⚡ Quick Answer
Is it OpenAI Compatible? YES. It uses standard OpenAI-compatible endpoints (/v1/chat/completions and /v1/models).
Does it support Images and Text? YES. The UnslothProAPI node has a mode switch for Vision (Image + Text) and Text (Text only).

1. Overview
This plugin connects ComfyUI to your local Unsloth Studio instance. It allows you to run large language models (LLMs) and vision-language models (VLMs) directly from your workflow without leaving the ComfyUI interface.



<img width="532" height="1003" alt="Screenshot 2026-05-27 074133" src="https://github.com/user-attachments/assets/78342ca1-a8e4-4c5e-a551-0d6618e0f998" />

Key Features:

Live Model Picker: Automatically detects models loaded in Unsloth Studio.
Vision & Text Modes: Switch between analyzing images and pure text chat.
OpenAI Compatible: Uses standard API structures, making it easy to integrate with other tools.
Reasoning Support: Optional "Thinking" mode for chain-of-thought outputs.
2. Prerequisites
Before using this plugin, ensure you have:

Unsloth Studio installed and running locally (default port 8888).
At least one model loaded in Unsloth Studio.
The plugin installed in your ComfyUI custom_nodes folder.
3. Node 1: 🔍 Unsloth Model Picker
This node queries your local Unsloth Studio to find available models and provides a dropdown list for easy selection.

Inputs:
api_url: The address of your Unsloth Studio (default: http://localhost:8888).
api_key: Your API key if you have one set in Studio (leave blank if none).
model_id: A dropdown list populated automatically from the API.
Outputs:
model_id: The selected model name.
api_url: The URL string (for wiring to other nodes).
api_key: The key string.
Usage Tip: Click "Refresh" in ComfyUI if the model list doesn't update after loading a new model in Studio.

4. Node 2: ✨ Unsloth API PRO
This is the main node that sends requests to the model and returns the response.

A. Connection Settings
api_url: Connect from UnslothModelPicker or type manually.
api_key: Connect from UnslothModelPicker or type manually.
B. Mode Selection
mode: Choose between:
vision: Requires an image input. The model will analyze the image and answer your prompt.
text: Text-only mode. No image input needed.
C. Prompts
system_prompt: Instructions for the model's behavior (e.g., "You are a helpful assistant").
user_prompt: The actual question or instruction for the model.
D. Sampling Parameters (Defaults tuned for Gemma 4 2B)
These control how the model generates text. You can adjust them to change creativity vs. accuracy.

temperature: Default 1.0. Higher = more creative; Lower = more deterministic.
top_p: Default 0.95. Controls vocabulary diversity.
top_k: Default 64. Limits the number of tokens considered.
repetition_penalty: Default 1.1. Helps prevent the model from repeating itself.
E. Advanced Options
enable_thinking: Boolean. If True, the node outputs a separate thinking string containing the model's internal reasoning (if supported by the model).
reasoning_effort: Select none, low, medium, or high to control how much effort the model puts into reasoning.
F. Outputs
response: The final text answer from the model.
thinking: The chain-of-thought (only populated if enable_thinking is True).
active_model: The name of the model actually used for this request.
5. Workflow Examples
Example 1: Text Chat
Add UnslothModelPicker. Connect api_url.
Add UnslothProAPI.
Wire model_id, api_url, and api_key from Picker to API node.
Set mode to text.
Type your question in user_prompt.
Connect response to a Preview Text node to see the answer.
Example 2: Image Analysis (Vision)
Add an image source (e.g., Load Image or Load Image Batch).
Add UnslothModelPicker. Connect api_url.
Add UnslothProAPI.
Wire model_id, api_url, and api_key from Picker to API node.
Set mode to vision.
Connect your image to the image input on the API node.
Type a question like "Describe this image" in user_prompt.
Connect response to a Preview Text node.
6. Troubleshooting



Error	Cause	Solution
"Cannot reach localhost:8888"	Unsloth Studio is not running.	Start your Unsloth Studio instance.
"401 Unauthorized"	Invalid or missing API key.	Check your API key in Unsloth Studio settings.
"404 — no model loaded"	Studio is running but empty.	Load a model in Unsloth Studio first.
"mode='vision' but no IMAGE"	Vision mode selected without image.	Connect an image node or switch mode to text.
Model list empty	API endpoint issue.	Click Refresh in ComfyUI menu. Check api_url.
7. OpenAI Compatibility Details
This plugin is fully compatible with the OpenAI API structure.

Endpoints Used:
GET /v1/models (List available models)
POST /v1/chat/completions (Send inference requests)
Payload Structure: Uses standard messages array with role and content.
Integration: You can theoretically use other OpenAI-compatible clients (like Ollama, LM Studio, or vLLM) with this node by changing the api_url, provided they support these endpoints.
