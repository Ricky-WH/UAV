import os
import numpy as np

# ========== 配置区域 ==========

# 原始标签文件夹
LABEL_DIR = "/root/autodl-fs/datasets/DUT-Anti-UAV/labels/val"          # 改成你的标签目录

# 新标签输出文件夹（会自动创建，原标签不会被改）
OUTPUT_LABEL_DIR = "labels_area_cls"

# 标签文件后缀
LABEL_EXT = ".txt"

# =============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def collect_all_areas(label_dir, label_ext=".txt"):
    """
    遍历所有标签文件，收集每个 bbox 的面积（YOLO: class_name cx cy w h）
    返回一个 numpy 数组 areas
    """
    areas = []

    for root, _, files in os.walk(label_dir):
        for fname in files:
            if not fname.endswith(label_ext):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    # 格式：class_name cx cy w h
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        continue

                    area = w * h           # 相对面积
                    if area > 0:
                        areas.append(area)

    if len(areas) == 0:
        return np.array([])
    return np.array(areas)


def compute_thresholds(areas):
    small_th = np.percentile(areas, 33)
    medium_th = np.percentile(areas, 66)
    return small_th, medium_th


def relabel_by_area(label_dir, out_dir, small_th, medium_th, label_ext=".txt"):
    """
    第二遍：根据面积阈值修改类别并保存到新目录
    small: 类别名不变
    medium: 类别改成 "1"
    large: 类别改成 "2"
    """
    cnt_small = cnt_medium = cnt_large = 0

    for root, _, files in os.walk(label_dir):
        for fname in files:
            if not fname.endswith(label_ext):
                continue

            in_path = os.path.join(root, fname)

            # 计算输出路径，保持原有子目录结构
            rel_path = os.path.relpath(in_path, label_dir)
            out_path = os.path.join(out_dir, rel_path)
            out_dirname = os.path.dirname(out_path)
            ensure_dir(out_dirname)

            new_lines = []

            with open(in_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.rstrip("\n")
                    if not raw.strip():
                        continue
                    parts = raw.split()
                    if len(parts) < 5:
                        # 不规范的行原样写回
                        new_lines.append(raw)
                        continue

                    # 原来的类别名（字符串）
                    orig_cls = parts[0]

                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                    except ValueError:
                        # 坏行就原样写回
                        new_lines.append(raw)
                        continue

                    area = w * h

                    # 根据面积划分 small / medium / large
                    if area < small_th:
                        new_cls = orig_cls    # small: 不变
                        cnt_small += 1
                    elif area < medium_th:
                        new_cls = "1"         # medium: 改成 "1"
                        cnt_medium += 1
                    else:
                        new_cls = "2"         # large: 改成 "2"
                        cnt_large += 1

                    # 保留后面额外字段（如果有）
                    extra = parts[5:]
                    new_line_parts = [new_cls,
                                      f"{cx:.6f}", f"{cy:.6f}",
                                      f"{w:.6f}", f"{h:.6f}"] + extra
                    new_line = " ".join(new_line_parts)
                    new_lines.append(new_line)

            # 写入新的标签文件
            with open(out_path, "w", encoding="utf-8") as f_out:
                for nl in new_lines:
                    f_out.write(nl + "\n")

    print("重新标注完成：")
    print("  small  (类别不变) 数量:", cnt_small)
    print("  medium (类别→ 1) 数量:", cnt_medium)
    print("  large  (类别→ 2) 数量:", cnt_large)
    print("新标签目录:", out_dir)


if __name__ == "__main__":
    print("第一步：统计所有 bbox 面积...")
    areas = collect_all_areas(LABEL_DIR, LABEL_EXT)
    if areas.size == 0:
        print("没有找到任何有效 bbox，请检查 LABEL_DIR 和格式。")
        raise SystemExit

    small_th, medium_th = compute_thresholds(areas)
    print("面积阈值：")
    print("  small_th  (33% 分位):", small_th)
    print("  medium_th (66% 分位):", medium_th)

    print("第二步：按面积修改类别并输出到新标签文件夹...")
    ensure_dir(OUTPUT_LABEL_DIR)
    relabel_by_area(LABEL_DIR, OUTPUT_LABEL_DIR, small_th, medium_th, LABEL_EXT)