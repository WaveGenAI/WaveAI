from transformers import MusicgenForConditionalGeneration

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

audio_values = model.generate(
    **unconditional_inputs, do_sample=True, max_new_tokens=256
)
