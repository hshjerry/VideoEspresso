import json
import os
from tqdm import tqdm
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import argparse
# fix seed
torch.manual_seed(0)

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Evaluate close end task')
parser.add_argument('-mp', '--model_path', default='', type=str, help='path of model')
parser.add_argument('-sn', '--save_path', default='', type=str, help='this will influence the saving path')
parser.add_argument('-we', '--with_evidence', default=False, type=bool, help='with evidence or not')
parser.add_argument('-jp', '--json_path', default='bench_final.json', type=str, help='path of json file')
args = parser.parse_args()


# Configure model path and device
model_path = args.model_path
save_path = args.save_path
device = 0

tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto", attn_implementation="flash_attention_2")
model.eval()

# Load and parse the JSON file
json_file_path = args.json_path
with open(json_file_path, "r") as f:
    data = json.load(f)

# Define the inference function
def run_inference(video_path, question):    
    #video input
    fps = 24
    gen_kwargs = {"do_sample": True, "temperature": 0.8, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 512}
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n"+question+"<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    vr = VideoReader(video_path, ctx=cpu(0))
    f = vr.get_avg_fps()
    duration = len(vr) / f
    total_frame_num = len(vr)
    frame_input = min(128, int(fps * duration)+1)
    print("num frames: ", total_frame_num)
    print("num input: ", frame_input)
    if total_frame_num == 0:
        return ""
    if duration >= 0.5:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, frame_input, dtype=int)
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, fps, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def clean_options(option):
    cleaned_option = option.split("):", 1)[-1].strip()
    return cleaned_option

# Process each entry and run inference
output_data = []
output_file_template = save_path
print(f"Save Path: {save_path}")

for i, item in enumerate(tqdm(data, desc='Evaluated on Video Benchmark')):
    item = data[i]
    video_path = item["video_path"]

    options = item["options"]
    options_prompt = ""
    option_list = ["\n(A) ","(B) ","(C) ","(D) "]
    for j, opt in enumerate(options):
        options_prompt += option_list[j] + clean_options(opt) + "\n"
    correct_answer = item["correct_answer"]
    evidence = item["evidence"]
    task = item['task']
    question = item['question']

    if args.with_evidence:
        final_query = f"Please finish the {task} task. Question: {question}. Your inference evidence is {evidence}. You have the following options: {options_prompt}. Select the answer and only give the option letters."
    else: 
        final_query = f"Please finish the {task} task. Question: {question}. You have the following options: {options_prompt}. Select the answer and only give the option letters."

    # Call the model for inference
    model_output = run_inference(video_path, final_query)
    
    # Record the model's output options in the item
    item["model_output"] = model_output
    output_data.append(item)
    
    # Save data every 10 entries
    if (i + 1) % 10 == 0:
        with open(output_file_template, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)    
        
# Save the remaining data at the end
with open(output_file_template, "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)