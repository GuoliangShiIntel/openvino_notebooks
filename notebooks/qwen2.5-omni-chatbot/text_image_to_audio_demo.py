import requests
from pathlib import Path
from notebook_utils import device_widget
from qwen2_5_omni_helper import OVQwen2_5OmniModel
from transformers import Qwen2_5OmniProcessor

model_id = "Qwen/Qwen2.5-Omni-7B"
# model_dir = Path(model_id.split("/")[-1])
model_dir = "/home/shig/Github/openvino_notebooks/notebooks/qwen2.5-omni-chatbot/Qwen2.5-Omni-7B"

thinker_device = device_widget(default="CPU", exclude=["NPU"], description="Thinker device")
talker_device = device_widget(default="CPU", exclude=["NPU"], description="Talker device")
token2wav_device = device_widget(default="CPU", exclude=["NPU"], description="Token2Wav device")

#######################################################

from qwen_omni_utils import process_mm_info
import soundfile as sf
import IPython
from transformers import TextStreamer
from PIL import Image
from io import BytesIO
from notebook_utils import download_file
from IPython.display import display

# Demo: text-image input and Audio output

image_path = Path("cat.png")

if not image_path.exists():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(BytesIO(requests.get(url).content))
    image.save(image_path)
else:
    image = Image.open(image_path)

if not Path("Trailer.wav").exists():
    download_file("https://voiceage.com/wbsamples/in_mono/Trailer.wav", "Trailer.wav")

audio = sf.read("Trailer.wav")

if not Path("coco.mp4").exists():
    download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4",
        filename="coco.mp4",
    )

print("Question:\nWhat is unusual on this picture?")
display(image)
display(IPython.display.Audio("Trailer.wav"))
display(IPython.display.Video("coco.mp4"))
print("Answer:")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is unusual on this picture?"},
            {"type": "audio", "audio": "Trailer.wav"},
            {"type": "image", "image": "cat.png"},
            {"type": "video", "video": "coco.mp4"},
        ],
    },
]

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(text)
# ['<|im_start|>system\n
# You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n
# <|im_start|>user\n
# <|vision_bos|><|IMAGE|><|vision_eos|>What is unusual on this picture?<|im_end|>\n
# <|im_start|>assistant\n']

# ['<|im_start|>system\n
# You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n
# <|im_start|>user\n<
# |vision_bos|><|IMAGE|><|vision_eos|>What is unusual on this picture?<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n
# <|im_start|>assistant\n']

# ['<|im_start|>system\n
# You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n
# <|im_start|>user\n
# <|vision_bos|><|IMAGE|><|vision_eos|>What is unusual on this picture?<|audio_bos|><|AUDIO|><|audio_eos|><|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n
# <|im_start|>assistant\n']

audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
print(f"audios data: {len(audios)}, type: {type(audios[0])}, shape: {audios[0].shape}")
# shape: (80001,)
print(f"images data: {len(images)}, type: {type(images[0])}, shape: {images[0].size if isinstance(images[0], Image.Image) else images[0].shape}")
# shape: (1008, 672)
print(f"videos data: {len(videos)}, type: {type(videos[0])}, shape: {videos[0].shape}")
# shape: torch.Size([18, 3, 364, 644])

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
print("=== inputs informations ===")
print(f"key values: {inputs.keys()}") # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
                                      # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'feature_attention_mask', 'input_features'])
                                      # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features'])

for key, value in inputs.items():
    print(f"\n{key} shape: {value.shape}")
    if hasattr(value, 'dtype'):
        print(f"{key} type: {value.dtype}")
    print(f"{key} value: {value}")

print("=========================")

ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device.value, talker_device=talker_device.value, token2wav_device=token2wav_device.value)

text_ids = ov_model.generate(
    **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=False, thinker_max_new_tokens=256
)
# text_ids, audio = ov_model.generate(
#     **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=True, thinker_max_new_tokens=256
# )

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Token IDs:", text_ids[0].tolist())  # 查看原始 token 序列
print("解码结果:", text[0])                # 对比参数不同时的输出差异

# sf.write(
#     "output.wav",
#     audio.reshape(-1).detach().cpu().numpy(),
#     samplerate=24000,
# )

# display(IPython.display.Audio("output.wav"))
