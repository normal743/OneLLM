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
    # ============ Debug 模式 ============
    if args.debug:
        print("\n" + "="*60)
        print("🐛 DEBUG MODE ENABLED")
        print("="*60)
        print("- Will use mock captures (fixed test values)")
        print("- Temperature forced to 0")
        print("- Max generation tokens: 10")
        print("- Response text: 'this is a test'")
        print("="*60 + "\n")
        
        # Debug模式下的简化流程
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 读取CSV
        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()  # 去掉列名空格
        df['image_name'] = df['image_name'].astype(str).str.strip()
        df['comment'] = df['comment'].astype(str).str.strip()
        df['comment_number'] = df['comment_number'].astype(int)  # 直接转为整数，不需要strip
        
        grouped = df.groupby('image_name').filter(lambda x: len(x) == 5)
        image_list = grouped['image_name'].unique().tolist()
        random.seed(args.seed)
        selected_images = random.sample(image_list, min(args.num_samples, len(image_list)))
        
        print(f"Selected {len(selected_images)} images for debug testing\n")
        
        # Debug captures
        debug_captures = create_debug_captures()
        debug_response = "this is a test"
        
        for img_idx, img_name in enumerate(tqdm(selected_images, desc="Overall Progress"), 1):
            print(f"\n{'='*50}")
            print(f"Processing Image {img_idx}/{len(selected_images)}: {img_name}")
            print(f"{'='*50}")
            
            img_base_name = os.path.splitext(img_name)[0]
            
            # 1. Image-only
            print(f"[Image-only] Capturing...", end=" ", flush=True)
            np.savez_compressed(
                os.path.join(args.output_dir, f"{img_base_name}.npz"),
                **debug_captures
            )
            print(f"✓ Saved to {img_base_name}.npz")
            
            # 2. Text-only (5 prompts)
            prompts_df = grouped[grouped['image_name'] == img_name]
            
            for _, row in prompts_df.iterrows():
                prompt_idx = int(row['comment_number'])  # 去掉空格
                prompt_text = row['comment']  # 去掉空格
                
                print(f"\n{'-'*50}")
                print(f"Prompt {prompt_idx}/4: {prompt_text[:40]}...")
                print(f"{'-'*50}")
                
                # 创建prompt文件夹
                prompt_dir = os.path.join(args.output_dir, f"{img_base_name}_prompt{prompt_idx}")
                os.makedirs(prompt_dir, exist_ok=True)
                
                # 生成10个token的captures
                for token_idx in range(10):
                    print(f"\rGenerating token {token_idx+1}/10: {debug_response[:token_idx+10]}", end="", flush=True)
                    np.savez_compressed(
                        os.path.join(prompt_dir, f"token_{token_idx:03d}.npz"),
                        **debug_captures
                    )
                
                print(f"\n✓ Generated 10 tokens")
                
                # 保存response
                with open(os.path.join(prompt_dir, "response.txt"), 'w') as f:
                    f.write(debug_response)
                print(f"✓ Saved captures to {img_base_name}_prompt{prompt_idx}/")
        
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
    df = pd.read_csv(args.csv_path)  # 去掉 delimiter='|'
    df.columns = df.columns.str.strip()  # 去掉列名空格
    df['image_name'] = df['image_name'].astype(str).str.strip()
    df['comment'] = df['comment'].astype(str).str.strip()  # 去掉空格
    df['comment_number'] = df['comment_number'].astype(int)  # ← 改成这样！

    grouped = df.groupby('image_name').filter(lambda x: len(x) == 5)
    image_list = grouped['image_name'].unique().tolist()

    # 直接使用预定义的选定图像列表（假设你有 selected_images.csv 或硬编码列表）
    # 示例：从文件读取或硬编码
    selected_images = image_list  # 如果所有都选定；否则替换为你的选定列表
    # 或者：selected_images = pd.read_csv('selected_100_images.csv')['image_name'].tolist()

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
        
        print(f"\n{'='*50}")
        print(f"Processing Image {img_idx}/{len(selected_images)}: {img_name}")
        print(f"{'='*50}")
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = T_random_resized_crop(image).unsqueeze(0).cuda().to(target_dtype)
        
        img_base_name = os.path.splitext(img_name)[0]
        
        # ========== 1. Image-only ==========
        print(f"[Image-only] Capturing...", end=" ", flush=True)
        model.llma.enable_capture = True
        model.llma.captures = {}
        
        # 使用空prompt只forward一次
        empty_tokens = torch.tensor([[tokenizer.bos_id]]).cuda()
        
        with torch.cuda.amp.autocast(dtype=target_dtype):
            with torch.inference_mode():
                _ = model.llma.forward_inference(empty_tokens, 0, image=image, modal='image')
        
        # 保存image-only captures
        np.savez_compressed(
            os.path.join(args.output_dir, f"{img_base_name}.npz"),
            **model.llma.captures
        )
        model.llma.enable_capture = False
        print(f"✓ Saved to {img_base_name}.npz")
        
        # ========== 2. Text-only (5 prompts) ==========
        prompts_df = grouped[grouped['image_name'] == img_name]
        
        for _, row in prompts_df.iterrows():
            prompt_idx = int(row['comment_number'])
            prompt_text = row['comment']
            
            print(f"\n{'-'*50}")
            print(f"Prompt {prompt_idx}/4: {prompt_text[:40]}...")
            print(f"{'-'*50}")
            
            # 准备prompt
            conv.messages = []
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()
            
            # Tokenize
            prompt_tokens = tokenizer.encode(prompt_str, bos=True, eos=False)
            max_prompt_size = model.llma.params.max_seq_len - args.max_gen_len
            prompt_tokens = prompt_tokens[-max_prompt_size:]
            prompt_size = len(prompt_tokens)
            
            total_len = min(model.llma.params.max_seq_len, args.max_gen_len + prompt_size)
            tokens = torch.full([total_len], tokenizer.pad_id).cuda().long()
            tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
            
            # 创建prompt文件夹
            prompt_dir = os.path.join(args.output_dir, f"{img_base_name}_prompt{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            # 手动生成循环
            start_pos = prompt_size
            prev_pos = 0
            generated_tokens = []
            
            for cur_pos in range(start_pos, total_len):
                # 清空并启用capture
                model.llma.captures = {}
                model.llma.enable_capture = True
                
                with torch.cuda.amp.autocast(dtype=target_dtype):
                    with torch.inference_mode():
                        logits = model.llma.forward_inference(
                            tokens[None, prev_pos:cur_pos], 
                            prev_pos, 
                            image=None,  # text-only
                            modal='image'
                        )
                
                # 保存当前token的captures
                token_save_idx = cur_pos - start_pos
                np.savez_compressed(
                    os.path.join(prompt_dir, f"token_{token_save_idx:03d}.npz"),
                    **model.llma.captures
                )
                model.llma.enable_capture = False
                
                # 采样下一个token
                if args.temperature > 0:
                    probs = torch.softmax(logits / args.temperature, dim=-1)
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    mask = probs_sum - probs_sort > args.top_p
                    probs_sort[mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token).item()
                else:
                    next_token = torch.argmax(logits, dim=-1).item()
                
                # 检查结束
                if next_token == tokenizer.eos_id:
                    break
                
                tokens[cur_pos] = next_token
                generated_tokens.append(next_token)
                prev_pos = cur_pos
                
                # 实时显示生成的文本
                current_text = tokenizer.decode(tokens[start_pos:cur_pos+1].tolist())
                print(f"\rGenerating: {current_text[:60]}...", end="", flush=True)
            
            print(f"\n✓ Generated {len(generated_tokens)} tokens")
            
            # 保存完整response
            final_text = tokenizer.decode(tokens[start_pos:start_pos+len(generated_tokens)].tolist())
            with open(os.path.join(prompt_dir, "response.txt"), 'w') as f:
                f.write(final_text)
            
            print(f"✓ Saved captures to {img_base_name}_prompt{prompt_idx}/ ({len(generated_tokens)} files)")
    
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
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p sampling (ignored if temperature=0)")
    parser.add_argument("--max_gen_len", type=int, default=256, help="maximum generation length")
    
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