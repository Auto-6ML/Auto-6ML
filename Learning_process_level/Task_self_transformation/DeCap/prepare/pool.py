import os
import pickle
import random
import json
import sys


def get_label_list(label_path):
    with open(label_path, "r") as file:
        label_list = [line.strip().replace("_", " ") for line in file]
    return label_list


def maybe_merge(path, name="blip_text"):
    if os.path.isdir(path):
        files = os.listdir(path)
        prompts = {}
        for file in files:
            if name in file:
                text_path = os.path.join(path, file)
                with open(text_path, "rb") as file:
                    prompt = pickle.load(file)
                for k, v in prompt.items():
                    if k not in prompts:
                        prompts[k] = v
                    else:
                        prompts[k] += v
    else:
        with open(path, "rb") as file:
            prompts = pickle.load(file)
    return prompts


# we use hand prompt provided in https://github.com/openai/CLIP/blob/main/data/prompts.md,
# if you are trying other domain datasets like EuroSAT, please substitude these prompts
hand_prompt = ['a good photo of the {}.', 'a photo of many {}.', 'a sculpture of a {}.',
               'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.',
               'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.',
               'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.',
               'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
               'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
               'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
               'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
               'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.',
               'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
               'a bad photo of a {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
               'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
               'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a {}.', 'a origami {}.',
               'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
               'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
               'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
               'a pixelated photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
               'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
               'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
               'a black and white photo of a {}.', 'a dark photo of a {}.', 'graffiti of the {}.', 'a toy {}.',
               'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'a digital style {}',
               'a colorful {}', 'a modern style {}', 'an abstract photo of {}', 'a cartoon style {}',
               'a virtual style {}', 'An ink painting of a {}', 'a toy {}', 'A model {}.', 'a red {}', 'a blue {}',
               'a yellow {}', 'a black {}', 'a white {}', 'An old {}.', 'A futuristic {}.', 'A minimalist {}.',
               'A detailed illustration of {}.', 'A close-up of {}.', 'A shadowy figure of {}.', 'A silhouette of {}.',
               'A bright and vibrant {}.', 'An abstract concept of {}.', 'A vintage style {}.', 'A neon-lit {}.',
               'A monochrome {}.', 'A watercolor painting of {}.', 'A sketch of {}.', 'A digital art of {}.',
               'A handcrafted {}.', 'An aerial view of {}.', 'A side profile of {}.', 'A textured {}.', 'A glossy {}.',
               'A matte {}.', 'A glowing {}.', 'A rustic {}.', 'A weathered {}.', 'A sparkling {}.', 'A serene {}.',
               'A chaotic {}.', 'A whimsical {}.', 'A dynamic {}.', 'A frozen moment of {}.', 'A soft-focus {}.',
               'A high-contrast {}.', 'A sepia-toned {}.', 'A saturated {}.', 'An isolated {}.', 'A mirrored {}.',
               'A panoramic view of {}.', 'An enchanted {}.']

dataset = sys.argv[1]

blip_text_path = './generated_prompt/{}_blip_text.pkl'.format(dataset)
llm_text_path = "./generated_prompt/{}_data_text.pkl".format(dataset)
label_path = './dataset/{}/cls_classes.txt'.format(dataset)
save_dir = './prompt_pool'
blip_prompt = maybe_merge(blip_text_path)
llm_prompt = maybe_merge(llm_text_path)
keys_ = list(blip_prompt.keys())
keys = list(llm_prompt.keys())
label_list = get_label_list(label_path)
# prompt_dict = {label: blip_prompt[label] + llm_prompt[label] for label in label_list}
prompt_dict = {}
lens = []
for i, label in enumerate(label_list):
    # class_prompt = blip_prompt[label] + llm_prompt[label]
    class_prompt = blip_prompt[keys_[i]] + llm_prompt[keys[i]]
    pool_prompt = [prompt.format(label) for prompt in hand_prompt]
    # prompt_dict[label] = pool_prompt + class_prompt
    prompt_dict[label] = class_prompt + pool_prompt
    # prompt_dict[label] = blip_prompt[keys_[i]] + llm_prompt[keys[i]]
    lens.append(len(class_prompt) + len(pool_prompt))
min_len = min(lens)
print("pool size:", min_len)
pool = {}
for k, v in prompt_dict.items():
    pool[k] = v[:min_len]
key1 = list(pool.keys())[0]
print(pool[key1])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, "{}_prompt_pool.pkl".format(dataset)), "wb") as file:
    pickle.dump(pool, file)
