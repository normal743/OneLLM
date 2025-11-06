import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
import argparse
import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import setup_for_distributed
from util.misc import default_tensor_type
from model.meta import MetaModel
from data.conversation_lib import conv_templates
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import random
from tqdm import tqdm


T_random_resized_crop = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=3,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def create_debug_captures():
    """创建固定的测试captures数据"""
    debug_data = {}
    # 模拟各种capture点
    debug_data['image_01_tokens'] = np.ones((1, 30, 512), dtype=np.float32) * 0.5
    debug_data['image_02_after_tokenizer'] = np.ones((1, 256, 1024), dtype=np.float32) * 0.5
    debug_data['text_01_tokens'] = np.ones((1, 10, 512), dtype=np.float32) * 0.5
    debug_data['combine_02_transformer_layer0'] = np.ones((1, 40, 512), dtype=np.float32) * 0.5
    debug_data['combine_03_final_output'] = np.ones((1, 512), dtype=np.float32) * 0.5
    return debug_data


def model_worker(args: argparse.Namespace) -> None:
    # ============ 参数验证 ============
    if args.max_gen_len != 1:
        raise ValueError(f"max_gen_len must be 1, got {args.max_gen_len}")
    
    # ============ Debug 模式 ============
    if args.debug:
        print("\n" + "="*60)
        print("🐛 DEBUG MODE ENABLED")
        print("="*60)
        print("- Will use mock captures (fixed test values)")
        print("- max_gen_len = 1 (enforced)")
        print("="*60 + "\n")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 读取CSV
        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        df['image_name'] = df['image_name'].astype(str).str.strip()
        df['comment'] = df['comment'].astype(str).str.strip()
        df['comment_number'] = df['comment_number'].astype(int)
        
        grouped = df.groupby('image_name').filter(lambda x: len(x) == 5)
        image_list = grouped['image_name'].unique().tolist()

        random.seed(args.seed)
        selected_images = random.sample(image_list, min(args.num_samples, len(image_list)))
        
        print(f'Selected {len(selected_images)} images for processing\n')
        
        debug_captures = create_debug_captures()
        
        for img_idx, img_name in enumerate(tqdm(selected_images, desc="Overall Progress"), 1):
            img_base_name = os.path.splitext(img_name)[0]
            
            # 创建图片文件夹
            img_folder = os.path.join(args.output_dir, img_base_name)
            os.makedirs(img_folder, exist_ok=True)
            
            print(f"\n{'='*50}")
            print(f"Processing {img_idx}/{len(selected_images)}: {img_name}")
            print(f"{'='*50}")
            
            # 1. Image-only
            print(f"[1/6] Saving image.npz...", end=" ", flush=True)
            image_path = os.path.join(img_folder, "image.npz")
            
            # 调试：打印路径和数据
            print(f"\n  DEBUG: Saving to {os.path.abspath(image_path)}")
            print(f"  DEBUG: debug_captures keys: {list(debug_captures.keys())}")
            print(f"  DEBUG: debug_captures size: {sum(v.nbytes for v in debug_captures.values())/1024:.1f} KB")
            
            np.savez_compressed(
                image_path,
                **debug_captures
            )
            
            # 验证文件存在
            file_exists = os.path.exists(image_path)
            file_size = os.path.getsize(image_path) if file_exists else 0
            print(f"  DEBUG: File exists={file_exists}, size={file_size/1024:.1f} KB")
            print("✓")
            
            # 2. Text-only (5 prompts)
            prompts_df = grouped[grouped['image_name'] == img_name]
            
            for _, row in prompts_df.iterrows():
                prompt_idx = int(row['comment_number'])
                print(f"[{prompt_idx+2}/6] Saving prompt{prompt_idx}.npz...", end=" ", flush=True)
                
                np.savez_compressed(
                    os.path.join(img_folder, f"prompt{prompt_idx}.npz"),
                    **debug_captures
                )
                print("✓")
            
            print(f"✓ Completed {img_base_name}/ (6 files)")
        
        print("\n" + "="*60)
        print("🐛 DEBUG MODE COMPLETED")
        print("="*60)
        return
    
    # ============ 正常模式 ============
    rank = 0
    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel(args.llama_type, args.llama_config, tokenizer_path=args.tokenizer_path)
    print("Loading pretrained weights ...")
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.cuda()
    model.eval()
    print(f"Model = {str(model)}")

    # 读取CSV数据
    print('Loading CSV data...')
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()
    df['image_name'] = df['image_name'].astype(str).str.strip()
    df['comment'] = df['comment'].astype(str).str.strip()
    df['comment_number'] = df['comment_number'].astype(int)

    grouped = df.groupby('image_name').filter(lambda x: len(x) == 5)
    image_list = grouped['image_name'].unique().tolist()

    random.seed(args.seed)
    selected_images = random.sample(image_list, min(args.num_samples, len(image_list)))
    
    print(f'Selected {len(selected_images)} images for processing\n')
    
    os.makedirs(args.output_dir, exist_ok=True)

    conv = conv_templates["v1"].copy()
    tokenizer = model.tokenizer

    # 主循环
    for img_idx, img_name in enumerate(tqdm(selected_images, desc="Overall Progress"), 1):
        img_path = os.path.join(args.image_folder, img_name)
        if not os.path.exists(img_path):
            print(f"\n⚠️  Image not found: {img_path}, skipping...")
            continue
        
        img_base_name = os.path.splitext(img_name)[0]
        
        # 创建图片文件夹
        img_folder = os.path.join(args.output_dir, img_base_name)
        os.makedirs(img_folder, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Processing {img_idx}/{len(selected_images)}: {img_name}")
        print(f"{'='*50}")
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = T_random_resized_crop(image).unsqueeze(0).cuda().to(target_dtype)
        
        # ========== 1. Image-only ==========
        print(f"[1/6] Capturing image.npz...", end=" ", flush=True)
        model.llma.enable_capture = True
        model.llma.captures = {}
        
        empty_tokens = torch.tensor([[tokenizer.bos_id]]).cuda()
        
        with torch.cuda.amp.autocast(dtype=target_dtype):
            with torch.inference_mode():
                _ = model.llma.forward_inference(empty_tokens, 0, image=image, modal='image')
        
        np.savez_compressed(
            os.path.join(img_folder, "image.npz"),
            **model.llma.captures
        )
        model.llma.enable_capture = False
        print("✓")
        
        # ========== 2. Text-only (5 prompts) ==========
        prompts_df = grouped[grouped['image_name'] == img_name]
        
        for _, row in prompts_df.iterrows():
            prompt_idx = int(row['comment_number'])
            prompt_text = row['comment']
            
            print(f"[{prompt_idx+2}/6] Capturing prompt{prompt_idx}.npz: {prompt_text[:30]}...", end=" ", flush=True)
            
            # 准备prompt
            conv.messages = []
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()
            
            # Tokenize
            prompt_tokens = tokenizer.encode(prompt_str, bos=True, eos=False)
            tokens = torch.tensor(prompt_tokens).cuda().long().unsqueeze(0)
            
            # Capture
            model.llma.captures = {}
            model.llma.enable_capture = True
            
            with torch.cuda.amp.autocast(dtype=target_dtype):
                with torch.inference_mode():
                    _ = model.llma.forward_inference(
                        tokens, 
                        0, 
                        image=None,
                        modal='image'
                    )
            
            # 保存
            np.savez_compressed(
                os.path.join(img_folder, f"prompt{prompt_idx}.npz"),
                **model.llma.captures
            )
            model.llma.enable_capture = False
            print("✓")
        
        print(f"✓ Completed {img_base_name}/ (6 files)")
    
    print("\n" + "="*60)
    print("✅ All processing completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OneLLM Capture Tool")
    
    # 数据相关
    parser.add_argument("--csv_path", type=str, required=True, help="path to the CSV file")
    parser.add_argument("--image_folder", type=str, required=True, help="folder containing images")
    parser.add_argument("--output_dir", type=str, default="./captures_output", help="directory to save captures")
    
    # 采样相关
    parser.add_argument("--num_samples", type=int, default=100, help="number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    # 生成参数 (固定为1)
    parser.add_argument("--max_gen_len", type=int, default=1, help="maximum generation length (must be 1)")
    
    # Debug模式
    parser.add_argument("--debug", action="store_true", help="enable debug mode with mock captures")
    
    # 模型相关
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=None,
        help="A list of space-separated gpu ids to run the model on."
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="config/llama2/tokenizer.model",
        help="Path to the tokenizer.model file."
    )
    parser.add_argument(
        "--llama_type", default="onellm", type=str,
        help="LLaMA model type."
    )
    parser.add_argument(
        "--llama_config", type=str, required=True,
        help="Path to the llama model config json."
    )
    parser.add_argument(
        "--pretrained_path", type=str,
        help="Path to the llama model checkpoints."
    )
    parser.add_argument(
        "--master_port", type=int, default=23862,
        help="Port for PyTorch distributed."
    )
    parser.add_argument(
        "--master_addr", type=str, default="127.0.0.1",
        help="Address for PyTorch distributed."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16"], default="fp16",
        help="Data type for model weights."
    )
    
    args = parser.parse_args()

    # GPU设置
    if args.gpu_ids is None:
        args.gpu_ids = [0]

    # Debug模式不需要pretrained_path
    if not args.debug and args.pretrained_path is None:
        parser.error("--pretrained_path is required in normal mode")

    mp.set_start_method("spawn")
    model_worker(args)