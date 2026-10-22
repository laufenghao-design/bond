import json
import csv
import os
import numpy as np
from tqdm import tqdm

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


def process_jsonl_to_csv(file_path, output_dir=None):
    """
    读取jsonl，计算每层的平均值、方差、标准差，并保存为CSV。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return

    # 1. 准备路径和文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = output_dir or os.path.join(os.path.dirname(file_path), "csv_output")
    # output_dir = os.path.join(os.path.dirname(file_path), "csv_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Target File: {base_name}")
    print(f"{'=' * 60}")

    # 2. 预先统计总行数
    print("Pre-scanning file to count lines...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 存储结构: layer_idx -> { 'sum': np.array, 'sum_sq': np.array, 'count': int }
    layer_data = {}

    # 3. 读取并累加数据
    with open(file_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(f, total=total_lines, unit="sample", dynamic_ncols=True)

        for i, line in enumerate(pbar):
            pbar.set_description(f"Reading {base_name[:15]}...")

            try:
                item = json.loads(line)
                layer_idx = item['layer_idx']

                rvt = item['relation_vis_text']
                if isinstance(rvt[0], list):
                    values = np.array(rvt[0], dtype=np.float64)
                else:
                    values = np.array(rvt, dtype=np.float64)

                if layer_idx not in layer_data:
                    # 初始化 sum 和 sum_sq (平方和)
                    layer_data[layer_idx] = {
                        'sum': np.zeros_like(values),
                        'sum_sq': np.zeros_like(values),  # 新增：存储平方和
                        'count': 0,
                    }

                # 累加值
                layer_data[layer_idx]['sum'] += values
                # 累加值的平方
                layer_data[layer_idx]['sum_sq'] += values**2

                layer_data[layer_idx]['count'] += 1

                pbar.set_postfix({"Layer": layer_idx, "Sample_Idx": i})

            except json.JSONDecodeError:
                print(f"\nSkipping invalid JSON at line {i}")
                continue
            except Exception as e:
                print(f"\nError processing line {i}: {e}")
                continue

    # 4. 计算统计量并写入 CSV
    print(f"\nComputing stats (Mean, Var, Std) and writing CSVs to: {output_dir}")

    sorted_layers = sorted(layer_data.keys())

    for layer_idx in tqdm(sorted_layers, desc="Writing CSV Layers", unit="layer"):
        data = layer_data[layer_idx]
        N = data['count']

        if N == 0:
            continue

        # --- 核心计算逻辑 ---
        # 1. 计算均值 E[X]
        mean_values = data['sum'] / N

        # 2. 计算平方的均值 E[X^2]
        mean_sq_values = data['sum_sq'] / N

        # 3. 计算方差 Var = E[X^2] - (E[X])^2
        # 使用 np.maximum 避免浮点数精度问题导致出现极小的负数（如 -1e-18）
        variance_values = np.maximum(mean_sq_values - (mean_values**2), 0)

        # 4. 计算标准差 Std = sqrt(Var)
        std_values = np.sqrt(variance_values)
        # -------------------

        # 构造文件名
        csv_filename = f"relation_vis_text_stats_{base_name}_layer{layer_idx}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 更新表头，包含 mean, variance, std_dev
            fieldnames = ['mean', 'variance', 'std_dev']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 使用 zip 同时遍历三个数组
            for m, v, s in zip(mean_values, variance_values, std_values):
                writer.writerow({'mean': m, 'variance': v, 'std_dev': s})

    print(f"Finished {base_name}. Generated {len(layer_data)} CSV files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate recorded reference scores by layer")
    parser.add_argument("files", nargs="+", help="Raw reference JSONL files")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for file_path in args.files:
        process_jsonl_to_csv(file_path, args.output_dir)
