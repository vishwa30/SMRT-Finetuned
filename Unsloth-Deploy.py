from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

# import ollama
import json
from fastapi import FastAPI
import pandas as pd
from PIL import Image
import io

import glob
import PIL
from matplotlib import colors
import pandas as pd

app = FastAPI()

# Temporary directory to save uploaded images  
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



import uvicorn
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
# model_name = "lora_model-1", # YOUR MODEL YOU USED FOR TRAINING
model_name = r"C:\Users\Administrator\Desktop\Codes\Unsloth\lora_model-2\lora_model-2",
load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!


def parse_attribute_string(attr_str):
    result = {}
    for part in attr_str.split(", "):
        if ": " in part:
            key, value = part.split(": ", 1)
            result[key.strip()] = value.strip()
    return result

def predicted_result(file_path):
    img = PIL.Image.open(file_path)
    # image = img.resize((300, 300))
    # image = img.resize((600,600))
    image = img
    instruction = """
    You are an expert in cloth identification. Answer in JSON format of primary picture in the image precisely.
    JSON format should be like: {'Item Type': 'Shorts', 'Colors': ['Orange', 'Green'], 'Patterns': ['Floral', 'Solid'], 'Fabric': 'Cotton', 'Description': 'The shirt is round collared and has fade in bottom.', 'Obvious Stain': 'No'}.
    Please avoid changing the JSON structure and key names.
    """

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")



    # Get input length in tokens
    input_len = inputs["input_ids"].shape[1]

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        use_cache=True,
        temperature=0.7,
        min_p=0.1,
    )

    # Slice only the new tokens (generated ones)
    new_tokens = output_ids[0][input_len:]

    # Decode only the new tokens
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(generated_text)
    pred_json = parse_attribute_string(generated_text)
    # for key,value in pred_json.items():
    #     if key == 'Item Type':
    #         pred_json['Type'] = pred_json.pop('Item Type')
    #     if key == 'Pattern':
    #         pred_json['Patterns'] = pred_json.pop('Pattern')
    print(pred_json)
    return(pred_json)




@app.post("/finetune-v1/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to disk
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_name_wo = (file.filename).split('.')[0]
        print("File name ", file_name_wo)
        # gt_json = ground_truth(file_name_wo)
        pred_json = predicted_result(file_path)

        print(pred_json)
        # return JSONResponse(content={
        #     "Prediction":pred_json
        # })
        # return pred_json
        return JSONResponse(content=pred_json)
           

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Run the FastAPI app with Uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
