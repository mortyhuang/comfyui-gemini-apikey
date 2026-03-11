import torch
import json
import base64
import requests
import numpy as np
from PIL import Image
import io
import os

# --- Configuration Loading Logic ---
def get_api_key():
    """
    Search for Gemini API Key in:
    1. Environment variable: GEMINI_API_KEY
    2. ComfyUI Global User Config: ComfyUI/user/gemini_config.json
    3. Local Node Config: custom_nodes/comfyui-gemini-imagen/config.json
    """
    # 1. Env Var
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    # 2. Global User Config (Persistent across updates)
    try:
        from folder_paths import base_path
        user_config_path = os.path.join(base_path, "user", "gemini_config.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, 'r') as f:
                config = json.load(f)
                return config.get("api_key", "")
    except:
        pass

    # 3. Local Node Config
    local_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, 'r') as f:
                config = json.load(f)
                return config.get("api_key", "")
        except:
            pass
            
    return ""

GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

class GeminiNanoBanana2_APIKey:
    """
    Nano Banana 2 with hidden API Key (loaded from config).
    Designed for clean UI and ComfyUI-Manager compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Text prompt describing the image."
                }),
                "model": (["Nano Banana 2 (Gemini 3.1 Flash Image)"], {"default": "Nano Banana 2 (Gemini 3.1 Flash Image)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "aspect_ratio": ([
                    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {"default": "auto"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"], {"default": "IMAGE"}),
                "thinking_level": (["MINIMAL", "HIGH"], {"default": "MINIMAL"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional reference image(s). To include multiple images, use the Batch Images node (up to 14)."}),
                "files": ("GEMINI_INPUT_FILES", {"tooltip": "Optional file(s) from Gemini Input Files node."}),
                "system_prompt": ("STRING", {"default": GEMINI_IMAGE_SYS_PROMPT, "multiline": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "STRING")
    FUNCTION = "execute"
    CATEGORY = "api node/image/Gemini"

    def tensor_to_base64(self, tensor):
        b64_list = []
        for i in range(tensor.shape[0]):
            img_np = 255. * tensor[i].cpu().numpy()
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        return b64_list

    def execute(self, prompt, model, seed, aspect_ratio, resolution, response_modalities, thinking_level, images=None, files=None, system_prompt=""):
        # API key is now strictly loaded from persistent storage
        actual_key = get_api_key()
        if not actual_key:
            raise ValueError(
                "Gemini API Key missing!\n"
                "Please create a file at: ComfyUI/user/gemini_config.json\n"
                "With the following content:\n"
                "{\n  \"api_key\": \"YOUR_ACTUAL_KEY_HERE\"\n}"
            )

        gemini_model = "gemini-3.1-flash-image-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": actual_key
        }

        # Build request parts
        parts = [{"text": prompt}]
        if images is not None:
            b64_images = self.tensor_to_base64(images)
            for b64 in b64_images:
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        
        if files is not None:
            for file_part in files:
                parts.append(file_part)

        image_config = {"imageSize": resolution}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": (["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                "imageConfig": image_config,
                "thinkingConfig": {"thinkingLevel": thinking_level},
                "seed": seed
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Gemini API Error {response.status_code}: {response.text}")

        data = response.json()
        out_images = []
        out_text = ""

        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts_resp = content.get("parts", [])
            for part in parts_resp:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data:
                    img_data = inline_data.get("data")
                    if img_data:
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np)[None, ...]
                        out_images.append(img_tensor)
                if "text" in part:
                    out_text += part["text"] + "\n"

        if not out_images:
            reason = candidate.get("finishReason") if "candidates" in data else "UNKNOWN"
            raise Exception(f"No image was returned. Finish Reason: {reason}")

        final_image = torch.cat(out_images, dim=0) if len(out_images) > 1 else out_images[0]
        return (final_image, out_text.strip())

NODE_CLASS_MAPPINGS = {"GeminiNanoBanana2_APIKey": GeminiNanoBanana2_APIKey}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiNanoBanana2_APIKey": "Nano Banana 2 (API Key)"}
