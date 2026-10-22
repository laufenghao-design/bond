import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np

# Simplex noise generation.
try:
    from opensimplex import OpenSimplex
except ImportError:
    raise ImportError("Please install opensimplex first: pip install opensimplex")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model_testFull.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import logging
import datetime
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Generate probes with the original Simplex noise distribution.
def generate_simplex_noise_image(seed, size=(336, 336), scale=16.0):
    """
    Generate a Simplex noise image for CLIP encoder (336x336 pixels).

    Args:
        seed: Random seed.
        size: Image size (width, height).
        scale: Controls the 'zoom' level of the noise. Higher value = larger blobs.
               Gaussian noise works at pixel level, but Simplex is smooth,
               so we need a scale to make it visible as textures.
    """
    env_scale = os.getenv("NOISE_SCALE")
    if env_scale is not None:
        try:
            scale = float(env_scale)
        except ValueError:
            print(f"Warning: Invalid NOISE_SCALE '{env_scale}', using default {scale}")

    gen = OpenSimplex(seed=seed)
    w, h = size

    # 初始化空白图像矩阵
    noise = np.zeros((h, w, 3), dtype=np.uint8)

    # 遍历像素生成噪声 (Simplex 噪声通常需要在空间上连续采样)
    # 使用不同的 z-offset (0, 10, 20) 来让 RGB 三个通道不相关，产生彩色噪声
    for y in range(h):
        for x in range(w):
            # 将坐标缩放以控制频率
            nx = x / scale
            ny = y / scale

            # 生成 [-1, 1] 范围的值
            r = gen.noise3(nx, ny, 0.0)
            g = gen.noise3(nx, ny, 10.0)
            b = gen.noise3(nx, ny, 20.0)

            # 将 [-1, 1] 映射到 [0, 255]
            noise[y, x, 0] = int((r + 1) * 127.5)
            noise[y, x, 1] = int((g + 1) * 127.5)
            noise[y, x, 2] = int((b + 1) * 127.5)

    return Image.fromarray(noise, 'RGB')


class NoiseDataset(Dataset):
    def __init__(self, num_images, tokenizer, image_processor, model_config, prompt_text):
        self.num_images = num_images
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt_text = prompt_text

    def __getitem__(self, index):
        qs = self.prompt_text
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Build the probe for this sample seed.
        image = generate_simplex_noise_image(seed=index)

        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return self.num_images


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(num_images, tokenizer, image_processor, model_config, prompt_text, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = NoiseDataset(num_images, tokenizer, image_processor, model_config, prompt_text)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    return data_loader


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.'
        )

    data_loader = create_data_loader(args.num_images, tokenizer, image_processor, model.config, args.prompt_text)

    retained_tokens = args.retained_tokens
    for idx, (input_ids, image_tensor, image_sizes) in enumerate(tqdm(data_loader, total=args.num_images)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                retained_tokens=retained_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": args.prompt_text,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {"noise_image": True},
                }
            )
            + "\n"
        )
    ans_file.close()


if __name__ == "__main__":
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=100, help="Number of noise images to test")
    parser.add_argument(
        "--prompt-text", type=str, default="Describe this image.", help="Text prompt for image description"
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--retained_tokens", type=int, default=192)
    args = parser.parse_args()

    eval_model(args)
