import os

import torch
from PIL import Image
import math
import pickle
import sys
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, \
    AutoPipelineForText2Image, AutoPipelineForImage2Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration


class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, train_lines, processor):
        self.train_lines = train_lines
        self.processor = processor

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, idx):
        annotation_path = self.train_lines[idx].split(';')[1].strip()
        image = Image.open(annotation_path)
        encoding = self.processor(images=image, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        y = int(self.train_lines[idx].split(';')[0])
        return encoding, y


def load_blip(model_id="", device="cuda"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)
    print("Successfully load Blip")
    return processor, model


def pretrained_model_download(model_id, img2img=False):
    if img2img:
        if "turbo" or "xl" in model_id:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        if "turbo" or "xl" in model_id:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda")
    return pipe


def get_inputs_prompt(prompt, batch_size=1, height=512, width=512, scaleguide=0, image=None):
    num_inference_steps = 2
    if image is not None:
        return {"prompt": prompt, "guidance_scale": scaleguide, "num_inference_steps": num_inference_steps,
                "image": image}
    else:
        return {"prompt": prompt, "height": height, "width": width, "guidance_scale": scaleguide,
                "num_inference_steps": num_inference_steps}


def get_inputs_prompt_embs(prompt_embeds, batch_size=1, height=512, width=512, scaleguide=0):
    num_inference_steps = 2

    return {"prompt_embeds": prompt_embeds, "height": height, "width": width,
            "guidance_scale": scaleguide, "num_inference_steps": num_inference_steps}


def get_img(pipe, img_shape, pic_num=1, prompt_embeds=None, prompt=None, scaleguide=6, image=None):
    if isinstance(img_shape, int):
        width = height = img_shape
    elif isinstance(img_shape, tuple):
        if len(img_shape) == 2:
            height = img_shape[0]
            width = img_shape[1]
        elif len(img_shape) == 3:
            height = img_shape[1]
            width = img_shape[2]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    if prompt_embeds is not None:
        images = pipe(**get_inputs_prompt_embs(prompt_embeds=prompt_embeds, batch_size=pic_num, height=height,
                                               width=width, scaleguide=scaleguide)).images
    else:
        images = pipe(
            **get_inputs_prompt(prompt, batch_size=pic_num, height=height, width=width, scaleguide=scaleguide,
                                image=image)).images
    return images


def get_prompt(length, Blip, data, processor, device="cuda"):
    pbar = tqdm(total=length, desc="image caption process", postfix=dict, mininterval=0.3)
    prompt = []
    pixel_values = data["pixel_values"].to(device)
    for i in range(length):
        sentence = Blip.generate(pixel_values=pixel_values[i].unsqueeze(0))
        prompt.append(processor.decode(sentence[0], skip_special_tokens=True))
        pbar.update(1)
    pbar.close()
    return prompt


dataset = sys.argv[1]
model_id = "Salesforce/blip-image-captioning-base"
meta_data_path = './dataset/{}/cls_val.txt'.format(dataset)
label_path = './dataset/{}/cls_classes.txt'.format(dataset)
save_path = './generated_prompt/{}_blip_text.pkl'.format(dataset)


with open(label_path, 'r') as file:
    label_list = [line.strip().replace("_", " ") for line in file]
processor, BLIP = load_blip(model_id)
with open(meta_data_path, encoding='utf-8') as f:
    train_lines = f.readlines()
train_dataset = ImageCaptioningDataset(train_lines, processor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, pin_memory=True)
train_x, train_y = next(iter(train_loader))

prompt = get_prompt(len(train_dataset), BLIP, train_x, processor)

prompt_dict = {i: [] for i in label_list}
for i in range(train_y.shape[0]):
    prompt_dict[label_list[int(train_y[i])]].append(label_list[int(train_y[i])] + ", " + prompt[i])
r = open(save_path, "wb")
pickle.dump(prompt_dict, r)
r.close()
