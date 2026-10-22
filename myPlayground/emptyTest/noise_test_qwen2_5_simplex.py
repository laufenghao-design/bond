import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import sys

sys.path.append('./Qwen2_5-VL/')
from Qwen2_5VL.modeling_qwen2_5_vl_self_full_test import Qwen2_5_VLForConditionalGeneration

from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import logging
import datetime
from PIL import Image
import math
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import opensimplex library
try:
    from opensimplex import OpenSimplex
except ImportError:
    raise ImportError("Please install opensimplex first: pip install opensimplex")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _generate_single_image(args):
    """Worker function to generate and save a single image."""
    seed, save_dir, size, scale = args
    image = generate_simplex_noise_image(seed, size, scale)
    image_path = os.path.join(save_dir, f"simplex_{seed:05d}.png")
    image.save(image_path)
    return seed


def generate_and_save_images(num_images, save_dir, size=(640, 640), scale=16.0):
    """Generate and save Simplex noise images to disk using multi-process parallelism."""
    os.makedirs(save_dir, exist_ok=True)

    env_scale = os.getenv("NOISE_SCALE")
    if env_scale is not None:
        try:
            scale = float(env_scale)
        except ValueError:
            print(f"Warning: Invalid NOISE_SCALE '{env_scale}', using default {scale}")

    # Configure the number of probe-generation workers.
    cpu_count = os.cpu_count() or 1
    max_workers = int(os.environ.get("NOISE_WORKERS", "1"))

    print(f"Generating {num_images} Simplex noise images using {max_workers} workers...")

    # Prepare arguments for each worker
    tasks = [(seed, save_dir, size, scale) for seed in range(num_images)]

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track with tqdm
        futures = [executor.submit(_generate_single_image, task) for task in tasks]

        # Process completed tasks with progress bar
        for _ in tqdm(as_completed(futures), total=num_images):
            pass

    print(f"All images saved to {save_dir}")


def generate_simplex_noise_image(seed, size=(640, 640), scale=16.0):
    """
    Generate a Simplex noise image for vision encoder (1344x1344 pixels).

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

    # Initialize empty image matrix
    noise = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate noise by iterating through pixels
    # Use different z-offsets (0, 10, 20) to make RGB channels uncorrelated, producing color noise
    for y in range(h):
        for x in range(w):
            # Scale coordinates to control frequency
            nx = x / scale
            ny = y / scale

            # Generate values in [-1, 1] range
            r = gen.noise3(nx, ny, 0.0)
            g = gen.noise3(nx, ny, 10.0)
            b = gen.noise3(nx, ny, 20.0)

            # Map [-1, 1] to [0, 255]
            noise[y, x, 0] = int((r + 1) * 127.5)
            noise[y, x, 1] = int((g + 1) * 127.5)
            noise[y, x, 2] = int((b + 1) * 127.5)

    return Image.fromarray(noise, 'RGB')


class NoiseDataset(Dataset):
    def __init__(self, image_dir, processor, prompt_text):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "simplex_*.png")))
        self.processor = processor
        self.prompt_text = prompt_text
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": self.prompt_text}],
            },
        ]

        return messages, image.size

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch):
    messages_list, image_sizes = zip(*batch)
    return list(messages_list), list(image_sizes)


def create_data_loader(image_dir, processor, prompt_text, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = NoiseDataset(image_dir, processor, prompt_text)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    return data_loader


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(args.image_dir, processor, args.prompt_text)

    for idx, (messages_batch, image_sizes) in enumerate(tqdm(data_loader, total=len(data_loader))):
        messages = messages_batch[0]

        # Process messages
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Extract images from messages
        image_inputs = []
        for msg in messages:
            if msg["role"] == "user":
                for content_item in msg["content"]:
                    if content_item["type"] == "image":
                        image_inputs.append(content_item["image"])

        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": args.prompt_text,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_path,
                    "metadata": {"noise_image": True, "noise_type": "simplex"},
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
    parser.add_argument(
        "--mode", type=str, choices=["generate", "eval"], required=True, help="Mode: generate images or evaluate"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./myPlayground/emptyTest/simplex_image_Qwen",
        help="Directory to save/load images",
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of Simplex noise images to generate")
    parser.add_argument("--image-size", type=int, default=640, help="Image size (width and height)")
    parser.add_argument(
        "--prompt-text", type=str, default="Describe this image.", help="Text prompt for image description"
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1605632)
    args = parser.parse_args()

    # Construct image directory using the environment variable NOISE_SCALE
    env_scale = os.getenv("NOISE_SCALE")
    scale = float(env_scale) if env_scale is not None else 16.0
    args.image_dir = f"{args.image_dir}_scale_{scale}"

    if args.mode == "generate":
        generate_and_save_images(args.num_images, args.image_dir, size=(args.image_size, args.image_size), scale=scale)
    elif args.mode == "eval":
        eval_model(args)
