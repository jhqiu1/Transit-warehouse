import os
import shutil


def manage_directory(path):
    """
    检查指定的文件夹是否存在，如果存在则删除，然后重新创建。
    如果文件夹不存在，则直接创建。

    :param path: 文件夹路径
    """
    if os.path.exists(path):
        # 如果文件夹存在，先删除
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")

    # 重新创建文件夹
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")
