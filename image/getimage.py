import pandas as pd
import os
import shutil

# 读取选定的100个图像的CSV
selected_csv_file = '/Users/mannormal/Downloads/my research/origin/OneLLM/image/results.csv'
selected_data = pd.read_csv(selected_csv_file)
selected_images = selected_data['image_name'].unique().tolist()  # 获取唯一图像名称列表

# 原始图像路径
images_path = '/Users/mannormal/Downloads/my research/origin/OneLLM/image_large/flickr30k_images'

# 新文件夹路径，用于存放选定的图像
output_dir = '/Users/mannormal/Downloads/my research/origin/OneLLM/image/flickr30k_images'
os.makedirs(output_dir, exist_ok=True)

# 复制选定的图像
for img_name in selected_images:
    src_path = os.path.join(images_path, img_name)
    dst_path = os.path.join(output_dir, img_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)  # 复制文件，保留元数据
        print(f"Copied {img_name}")
    else:
        print(f"Image {img_name} not found")

print(f"Total selected images: {len(selected_images)}")