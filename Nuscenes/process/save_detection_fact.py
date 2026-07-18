import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import nuscenes_dir
from nuscenes.utils.geometry_utils import view_points

NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

OUTPUT_DIR = "./results_detection_fact/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for scene in nusc.scene[:100]:
    scene_idx = scene['name'][6:]  # 去掉前6个字符，只保留数字部分
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    cam_name = "CAM_FRONT"
    cam_data = nusc.get('sample_data', sample['data'][cam_name])
    image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])

    image = Image.open(image_path)
    image_np = np.array(image)
    img_width, img_height = image.size
    dpi = 100
    figsize = (img_width / dpi, img_height / dpi)

    _, boxes, cam_intrinsic = nusc.get_sample_data(sample['data'][cam_name])

    filtered_boxes = []
    removed_objects = []
    
    for box in boxes:
        ann = nusc.get('sample_annotation', box.token)
        visibility = int(ann['visibility_token'])
        if visibility >= 2:
            filtered_boxes.append(box)
        else:
            removed_objects.append(box.name)
    
    visible_objects = list(set([box.name for box in filtered_boxes]))
    fact_data = {
        "scene_name": scene['name'],
        "sample_token": first_sample_token,
        "visible_objects": visible_objects,
    }
    
    # **保存 JSON 文件**
    base_filename = f"{scene_idx}_{first_sample_token}"
    fact_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
    
    with open(fact_filename, 'w') as f:
        json.dump(fact_data, f, indent=2)
    print(f"Saved fact file: {fact_filename}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image_np)

    def draw_box3d(ax, box, cam_intrinsic):
        corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        for edge in edges:
            x_values = [corners_2d[0, edge[0]], corners_2d[0, edge[1]]]
            y_values = [corners_2d[1, edge[0]], corners_2d[1, edge[1]]]
            ax.plot(x_values, y_values, 'r-', linewidth=1.5)
        center_2d = view_points(box.center.reshape(3, 1), cam_intrinsic, normalize=True)[:2, 0]
        ax.text(center_2d[0], center_2d[1], box.name, color='blue', fontsize=8)
    
    for box in filtered_boxes:
        draw_box3d(ax, box, cam_intrinsic)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # **保存图像**
    output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.jpg")
    plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_filename}")

print("All images and fact files saved successfully.")
