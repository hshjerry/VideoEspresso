# Close-Ended Evaluation Guidance

---

## 1. Prerequisites

Before running the script, ensure the following dependencies are installed:

### Required Dependencies
- **Python Version**: 3.8 or higher
- **Libraries**:
  - `json`
  - `os`
  - `tqdm`
  - `PIL` (Pillow library)
  - `decord`
  - `torch`
  - `numpy`
  - `argparse`

### Installation
Use the following command to install most dependencies:

```bash
pip install tqdm pillow decord torch numpy
```

### Model-Specific Setup
The script uses **longva** as an example for model configuration. To use your own model, you must replace the **longva**-specific components with your own model loading and processing code (explained in Section 6).

---

## 2. Script Overview

The primary purpose of this script is to:
1. Load a pretrained video inference model.
2. Read input data from a specified JSON file.
3. Perform inference on each video and question.
4. Save inference results to a new JSON file.

---

## 3. Running the Script

### Command to Run
You need to specify key parameters such as the model path and JSON file path when running the script. Here's an example command:

```bash
python script.py -mp /path/to/your_model -sn /path/to/save.json -we True -jp /path/to/bench_final.json
```

### Parameter Details

| Parameter          | Short Form | Default Value          | Type    | Description                                                           |
|--------------------|------------|------------------------|---------|-----------------------------------------------------------------------|
| `--model_path`     | `-mp`      | Empty string (Required) | `str`   | Path to your model or checkpoint.                                     |
| `--save_path`      | `-sn`      | Empty string (Required) | `str`   | Path to save the inference results.                                   |
| `--with_evidence`  | `-we`      | `False`                | `bool`  | Whether to include evidence in the query.                            |
| `--json_path`      | `-jp`      | `bench_final.json`     | `str`   | Path to the input JSON file containing video data and questions.      |

---

## 4. Input JSON Structure

The input JSON file should contain data in the following format:

```json
[
    {
        "video_path": "path/to/video.mp4",
        "question": "What is happening in the video?",
        "options": [
            "(A): Option 1",
            "(B): Option 2",
            "(C): Option 3",
            "(D): Option 4"
        ],
        "correct_answer": "A",
        "evidence": "Some evidence information",
        "task": "Description of the task"
    }
]
```

### Field Descriptions
- **`video_path`**: Path to the video file.
- **`question`**: Question related to the video.
- **`options`**: Multiple-choice options.
- **`correct_answer`**: Correct answer (optional).
- **`evidence`**: Evidence to assist inference (optional).
- **`task`**: Description of the task.

---

## 5. Output Results

The script saves the results to the file specified by `--save_path`. The output JSON will include the model's predictions in the following format:

```json
[
    {
        "video_path": "path/to/video.mp4",
        "question": "What is happening in the video?",
        "options": [
            "(A): Option 1",
            "(B): Option 2",
            "(C): Option 3",
            "(D): Option 4"
        ],
        "correct_answer": "A",
        "evidence": "Some evidence information",
        "task": "Description of the task",
        "model_output": "A"
    }
]
```

---

## 6. Adapting the Model Configuration

The provided script uses **longva** as an example for model configuration. If you want to use your own model, follow these steps to adapt the script.

### Replace `longva` Model Loading
In the script, the following code is used to load the **longva** model:

```python
from longva.model.builder import load_pretrained_model

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_qwen", device_map="auto", attn_implementation="flash_attention_2"
)
model.eval()
```

To adapt this to your own model:
1. Import your model loading method (e.g., `from my_model_library import MyModel`).
2. Replace the `load_pretrained_model` function with your own logic to load the model, tokenizer, and any required preprocessing utilities.

For example:

```python
from my_model_library import MyModel, MyTokenizer, MyImageProcessor

# Load your model, tokenizer, and image processor
tokenizer = MyTokenizer.from_pretrained(model_path)
model = MyModel.from_pretrained(model_path).to("cuda")  # or "cpu"
image_processor = MyImageProcessor()
model.eval()
```

### Update Video Frame Preprocessing
The **longva** script uses an image processor for frames:

```python
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
```

To adapt this part:
- Replace `image_processor.preprocess` with your model's expected preprocessing steps.
- Ensure the video frames are converted into tensors compatible with your model.

For example:

```python
video_tensor = torch.tensor(frames).float().permute(0, 3, 1, 2)  # Convert frames to tensors
video_tensor = video_tensor.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to device
```

### Update Inference Logic
The **longva** script generates predictions with:

```python
output_ids = model.generate(input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
```

To adapt this:
- Replace `model.generate` with your model's inference method.
- Update how the output is decoded into text.

For example:

```python
# Perform inference using your model
with torch.no_grad():
    outputs = model(video_tensor, input_ids)

# Decode the output if necessary
decoded_output = tokenizer.decode(outputs, skip_special_tokens=True)
```

---

## 7. Key Functionalities in the Script

### 1. Video Frame Sampling
The script uses `decord` to load and uniformly sample frames from the video:

```python
vr = VideoReader(video_path, ctx=cpu(0))
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, frame_input, dtype=int)
frames = vr.get_batch(frame_idx).asnumpy()
```

Adapt this part if your model requires a different frame sampling strategy.

### 2. Inference Query Construction
The script dynamically constructs inference queries based on the input question, task, and options:

```python
final_query = f"Please finish the {task} task. Question: {question}. You have the following options: {options_prompt}. Select the answer and only give the option letters."
```

Customize this to align with your model's expected input format.

---

## 8. Notes

1. **Device Support**: The model defaults to using a GPU. If no GPU is available, the script will use a CPU, which may be slower.
2. **Video Format**: Ensure your video files are compatible with the `decord` library or your chosen video processing library.
3. **Error Handling**: If a video cannot be processed, the script may skip it. Add error-handling logic as needed.

---

## 9. Example Workflow

Here’s a general workflow for adapting this script to your own model:

1. **Replace Model Loading**: Update the imports and logic to load your custom model, tokenizer, and processor.
2. **Adapt Preprocessing**: Adjust the preprocessing pipeline for video frames to match your model's requirements.
3. **Modify Inference Logic**: Update the inference method and result decoding based on your model's API.
4. **Test the Script**: Run the script on sample data to ensure it works correctly with your model.

---

By following this guide, you can adapt the script to work with your own model and perform video-based inference seamlessly!
