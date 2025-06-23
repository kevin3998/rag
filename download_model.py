import os
from huggingface_hub import snapshot_download


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_id = "BAAI/bge-large-zh-v1.5"

print(f"开始使用镜像下载模型: {model_id}")
print("这个过程会下载约 1.2 GB 的文件，请耐心等待...")
print("如果中途断开，重新运行此脚本即可断点续传。")

try:
    # 使用 snapshot_download 下载整个模型仓库
    # 它会自动将文件保存到正确的本地缓存目录
    snapshot_download(
        repo_id=model_id,
        resume_download=True,  # 开启断点续传
        # 您可以指定一个本地目录来观察下载的文件，
        # 但为了让其他脚本自动找到，我们让它使用默认缓存位置
        # local_dir="./bge-large-zh-v1.5_model",
        # local_dir_use_symlinks=False
    )
    print("\n✅ 模型下载成功！")
    print("现在您可以运行 build_vectordb.py 脚本了。")

except Exception as e:
    print(f"\n❌ 下载过程中发生错误: {e}")
    print("请检查您的网络连接，并确认 HF_ENDPOINT 环境变量已正确设置。")