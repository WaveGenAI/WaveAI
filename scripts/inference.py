import argparse

import gradio as gr
import torch
from audiotools import AudioSignal

from waveai.audio_processor import AudioProcessor
from waveai.generation import Generation
from waveai.trainer import Trainer
from waveai.utils.config_parser import ConfigParser


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigParser(config_path=config_path)
    model = Trainer.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=config, audio_processor=None
    ).model
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, config, device


def setup_generation(model, config):
    audio_processor = AudioProcessor(config)
    generation = Generation(
        model,
        config.model.num_codebooks,
        config.model.pad_token_id,
        config.model.start_token_id,
        config.model.end_token_id,
        config.model.stereo,
    )
    return audio_processor, generation


@torch.no_grad()
def generate(src_text: str, duration, audio_processor, generation, device):
    prompt_embd, prompts_masks = audio_processor.encode_prompt([src_text])
    prompt_embd = prompt_embd.to(device)
    prompts_masks = prompts_masks.to(device)
    output_ids = generation.inference(prompt_embd, prompts_masks, duration=duration)
    y = audio_processor.decode_audio(output_ids)

    return y.audio_data, y.sample_rate


def gradio_generate(prompt, duration, device, audio_processor, generation):
    audio_data, sample_rate = generate(
        prompt, duration, audio_processor, generation, device
    )
    return (sample_rate, audio_data.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the Gradio interface"
    )
    args = parser.parse_args()

    model, config, device = load_model(args.config_path, args.checkpoint_path)
    audio_processor, generation = setup_generation(model, config)

    demo = gr.Interface(
        fn=lambda prompt, duration: gradio_generate(
            prompt,
            duration,
            device,
            audio_processor,
            generation,
        ),
        inputs=[
            gr.Textbox(lines=3, placeholder="Enter your prompt here..."),
            gr.Slider(minimum=1, maximum=30, step=1, label="Audio Duration (seconds)"),
        ],
        outputs=gr.Audio(type="numpy", label="Generated audio"),
        title="WaveAI Music Generator",
        description="Enter a text prompt and select the audio duration to generate a matching music clip.",
    )

    demo.launch(share=args.share)
