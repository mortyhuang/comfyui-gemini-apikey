# ComfyUI Gemini API Key Node

This node allows you to use Google's Gemini Nano Banana 2 (Imagen 3 / 3.1 Flash) models for image generation directly within ComfyUI using an API Key.

## Features
- Fully aligned with official Nano Banana 2 specifications.
- Clean UI: API Key is loaded from `ComfyUI/user/gemini_config.json`.
- Supports Multi-image input and official Gemini Input Files.
- Supports "Thinking" mode and System Prompts.

## Installation
1. Move this folder `comfyui-gemini-apikey` into your ComfyUI `custom_nodes` directory.
2. Create a file at `ComfyUI/user/gemini_config.json` with your key:
   ```json
   { "api_key": "YOUR_KEY_HERE" }
   ```
3. Restart ComfyUI.

## Parameters
- **prompt**: Text prompt for generation.
- **model**: Gemini 3.1 Flash Image (Nano Banana 2).
- **seed**: Seed for deterministic output.
- **aspect_ratio**: Support for 1:1, 16:9, 21:9, etc.
- **resolution**: 1K, 2K, 4K.
- **response_modalities**: IMAGE or IMAGE+TEXT.
- **thinking_level**: MINIMAL or HIGH.
