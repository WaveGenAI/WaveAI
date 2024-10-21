import argparse
import gradio as gr
import torch
from audiotools import AudioSignal
from transformers import AutoTokenizer
import waveai.text_encoder as text_encoder
from waveai.audio_autoencoder import DAC as audio_autoencoder
from waveai.generation import Generation
from waveai.trainer import Trainer
from waveai.utils.config_parser import ConfigParser


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigParser(config_path=config_path)
    model = Trainer.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=config, audio_processor=None
    ).model
    model.to(device)
    print(f"Loaded model from {checkpoint_path}")
    return model, config, device


def setup_generation(model, config, device):
    text_encoder_model = text_encoder.T5EncoderBaseModel()
    audio_codec = audio_autoencoder()
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
    generation = Generation(
        model,
        config.model.num_codebooks,
        config.model.pad_token_id,
        config.model.stereo,
    )
    return text_encoder_model, audio_codec, tokenizer, generation


@torch.no_grad()
def generate(
    src_text: str, duration, text_encoder_model, audio_codec, generation, device
):
    prompt_embd, prompts_masks = text_encoder_model([src_text])
    prompt_embd = prompt_embd.to(device)
    prompts_masks = prompts_masks.to(device)
    output_ids = generation.inference(prompt_embd, prompts_masks, duration=duration)
    y = audio_codec.decode(output_ids.cpu()).to(torch.float32)
    y = AudioSignal(y.cpu().numpy(), sample_rate=audio_codec.sample_rate)
    return y.audio_data, y.sample_rate


def gradio_generate(
    prompt, duration, device, text_encoder_model, audio_codec, generation
):
    audio_data, sample_rate = generate(
        prompt, duration, text_encoder_model, audio_codec, generation, device
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
    text_encoder_model, audio_codec, tokenizer, generation = setup_generation(
        model, config, device
    )

    demo = gr.Interface(
        fn=lambda prompt, duration: gradio_generate(
            prompt,
            duration,
            device,
            text_encoder_model,
            audio_codec,
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
