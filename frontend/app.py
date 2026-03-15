import json
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "models.json"
MODEL_CACHE: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
MODEL_STATUS: dict[str, dict[str, str]] = {}
STATUS_LOCK = threading.Lock()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def update_model_status(model_id: str, state: str, detail: str) -> None:
    with STATUS_LOCK:
        MODEL_STATUS[model_id] = {"state": state, "detail": detail}


def get_device_label() -> str:
    if torch.cuda.is_available():
        return "GPU"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Silicon"
    return "CPU"


def are_models_ready(config: dict[str, Any]) -> bool:
    model_ids = {model["id"] for model in config["models"]}
    with STATUS_LOCK:
        return all(MODEL_STATUS.get(model_id, {}).get("state") == "ready" for model_id in model_ids)


def get_status_view():
    config = load_config()
    lines = [
        "### Model status",
        f"Runtime device: **{get_device_label()}**",
        "",
    ]

    with STATUS_LOCK:
        for model in config["models"]:
            status = MODEL_STATUS.get(model["id"], {"state": "queued", "detail": "Waiting to start..."})
            lines.append(
                f"- **{model['label']}** (`{model['id']}`): `{status['state']}` - {status['detail']}"
            )

    ready = are_models_ready(config)
    button_text = "Compare models" if ready else "Loading models into memory..."
    return "\n".join(lines), gr.update(interactive=ready, value=button_text)


def get_model_bundle(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if model_id in MODEL_CACHE:
        update_model_status(model_id, "ready", "Already loaded in memory.")
        return MODEL_CACHE[model_id]

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "token": token,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    update_model_status(model_id, "downloading", "Fetching tokenizer and model weights...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    update_model_status(model_id, "loading", "Placing model weights into memory...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MODEL_CACHE[model_id] = (tokenizer, model)
    update_model_status(model_id, "ready", "Loaded in memory and ready for inference.")
    return tokenizer, model


def preload_models() -> None:
    config = load_config()
    model_ids = []

    for model in config["models"]:
        if model["id"] not in model_ids:
            model_ids.append(model["id"])

    for model_id in model_ids:
        try:
            get_model_bundle(model_id)
        except Exception as exc:
            update_model_status(model_id, "error", str(exc))


def start_preload_thread() -> None:
    config = load_config()
    with STATUS_LOCK:
        for model in config["models"]:
            MODEL_STATUS.setdefault(model["id"], {"state": "queued", "detail": "Waiting to start..."})

    threading.Thread(target=preload_models, daemon=True).start()


def call_model(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    tokenizer, model = get_model_bundle(model_id)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare_models(user_prompt: str, system_prompt: str, max_tokens: int, temperature: float):
    if not user_prompt.strip():
        raise gr.Error("Enter a prompt before running the comparison.")

    config = load_config()
    if not are_models_ready(config):
        raise gr.Error("Models are still loading into memory. Wait until the status shows ready.")

    model_a = config["models"][0]
    model_b = config["models"][1]
    response_a = call_model(
        model_a["id"],
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
    )
    response_b = call_model(
        model_b["id"],
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
    )

    model_info = (
        f"Model A: `{model_a['label']}` -> `{model_a['id']}`\n\n"
        f"Model B: `{model_b['label']}` -> `{model_b['id']}`"
    )
    return response_a, response_b, model_info


def build_app() -> gr.Blocks:
    config = load_config()
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    css = """
    .app-shell {
        max-width: 1180px;
        margin: 0 auto;
    }
    .hero {
        background: linear-gradient(135deg, #e6fffb 0%, #f8fafc 55%, #dbeafe 100%);
        border: 1px solid #cbd5e1;
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 18px;
    }
    .output-card {
        border: 1px solid #dbe4ee;
        border-radius: 20px;
    }
    """

    with gr.Blocks(theme=theme, css=css, title=config["app"]["title"]) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                f"""
                <div class="hero">
                  <div style="font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; color: #0f766e;">
                    Hugging Face model comparison
                  </div>
                  <h1 style="margin: 8px 0 10px; font-size: 2.1rem; color: #0f172a;">
                    {config["app"]["title"]}
                  </h1>
                  <p style="margin: 0; color: #334155; max-width: 760px;">
                    {config["app"]["description"]}
                  </p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=5):
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=8,
                        placeholder="Ask both models the same question...",
                    )
                with gr.Column(scale=3):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        lines=8,
                        value=config["defaults"]["system_prompt"],
                    )

            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max tokens",
                    minimum=64,
                    maximum=2048,
                    step=64,
                    value=config["defaults"]["max_tokens"],
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.1,
                    value=config["defaults"]["temperature"],
                )

            with gr.Row():
                run_button = gr.Button("Loading models into memory...", variant="primary", interactive=False)
                gr.ClearButton(value="Clear", components=[prompt, system_prompt])

            status_markdown = gr.Markdown()
            model_info = gr.Markdown(
                value=(
                    f"Model A: `{config['models'][0]['label']}` -> `{config['models'][0]['id']}`\n\n"
                    f"Model B: `{config['models'][1]['label']}` -> `{config['models'][1]['id']}`"
                )
            )

            with gr.Row():
                output_a = gr.Markdown(
                    label=config["models"][0]["label"],
                    elem_classes="output-card",
                    show_copy_button=True,
                )
                output_b = gr.Markdown(
                    label=config["models"][1]["label"],
                    elem_classes="output-card",
                    show_copy_button=True,
                )

            run_button.click(
                fn=compare_models,
                inputs=[prompt, system_prompt, max_tokens, temperature],
                outputs=[output_a, output_b, model_info],
                api_name="compare_models",
            )

            demo.load(fn=get_status_view, outputs=[status_markdown, run_button])
            gr.Timer(2).tick(fn=get_status_view, outputs=[status_markdown, run_button])

    return demo


start_preload_thread()
demo = build_app().queue(default_concurrency_limit=4)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
