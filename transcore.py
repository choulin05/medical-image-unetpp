import os
import shutil
from pathlib import Path

def extract_files_and_clean_empty_folders(root_folder):
    """
    处理三级文件夹结构：
    1. 将每个第二级文件夹中的所有第三级文件夹中的文件提取到相应的第二级文件夹中
    2. 删除所有空的第三级文件夹
    
    参数:
        root_folder: 第一级文件夹路径
    """
    root_path = Path(root_folder)
    
    # 检查第一级文件夹是否存在
    if not root_path.exists() or not root_path.is_dir():
        print(f"错误: 第一级文件夹 '{root_folder}' 不存在或不是一个文件夹")
        return
    
    # 获取所有第二级文件夹
    second_level_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not second_level_folders:
        print(f"第一级文件夹 '{root_folder}' 中没有第二级子文件夹")
        return
    
    print(f"找到 {len(second_level_folders)} 个第二级文件夹需要处理")
    print("=" * 60)
    
    total_files_moved = 0
    total_empty_folders_deleted = 0
    
    # 处理每个第二级文件夹
    for i, second_folder in enumerate(second_level_folders, 1):
        print(f"[{i}/{len(second_level_folders)}] 处理第二级文件夹: {second_folder.name}")
        
        # 获取第二级文件夹中的所有第三级文件夹
        third_level_folders = [f for f in second_folder.iterdir() if f.is_dir()]
        
        if not third_level_folders:
            print(f"  - 第二级文件夹 '{second_folder.name}' 中没有第三级子文件夹")
            continue
        
        files_moved_in_folder = 0
        empty_folders_deleted_in_folder = 0
        
        # 处理每个第三级文件夹
        for third_folder in third_level_folders:
            print(f"  - 处理第三级文件夹: {third_folder.name}")
            
            # 获取第三级文件夹中的所有文件
            files_in_third_folder = list(third_folder.iterdir())
            
            # 移动文件到第二级文件夹
            for file_path in files_in_third_folder:
                if file_path.is_file():
                    try:
                        # 处理文件名冲突
                        target_file_path = second_folder / file_path.name
                        counter = 1
                        while target_file_path.exists():
                            name, ext = os.path.splitext(file_path.name)
                            target_file_path = second_folder / f"{name}_{counter}{ext}"
                            counter += 1
                        
                        # 移动文件
                        shutil.move(str(file_path), str(target_file_path))
                        files_moved_in_folder += 1
                        print(f"    → 移动文件: {file_path.name} -> {target_file_path.name}")
                    except Exception as e:
                        print(f"    × 错误: 无法移动文件 {file_path.name} -> {e}")
            
            # 检查第三级文件夹是否为空，如果为空则删除
            remaining_items = list(third_folder.iterdir())
            if len(remaining_items) == 0:
                try:
                    third_folder.rmdir()
                    empty_folders_deleted_in_folder += 1
                    print(f"    ✓ 删除空文件夹: {third_folder.name}")
                except Exception as e:
                    print(f"    × 错误: 无法删除文件夹 {third_folder.name} -> {e}")
            else:
                print(f"    ! 文件夹 {third_folder.name} 不为空，保留")
        
        total_files_moved += files_moved_in_folder
        total_empty_folders_deleted += empty_folders_deleted_in_folder
        
        print(f"  - 完成! 移动了 {files_moved_in_folder} 个文件，删除了 {empty_folders_deleted_in_folder} 个空文件夹")
        print()
    
    print("=" * 60)
    print("所有操作完成!")
    print(f"总共处理了 {len(second_level_folders)} 个第二级文件夹")
    print(f"总共移动了 {total_files_moved} 个文件")
    print(f"总共删除了 {total_empty_folders_deleted} 个空文件夹")

def process_specific_second_level_folders(root_folder, specific_second_folders):
    """
    处理特定的第二级文件夹
    
    参数:
        root_folder: 第一级文件夹路径
        specific_second_folders: 要处理的特定第二级文件夹名称列表
    """
    root_path = Path(root_folder)
    
    # 检查第一级文件夹是否存在
    if not root_path.exists() or not root_path.is_dir():
        print(f"错误: 第一级文件夹 '{root_folder}' 不存在或不是一个文件夹")
        return
    
    total_files_moved = 0
    total_empty_folders_deleted = 0
    
    print(f"处理特定的第二级文件夹: {specific_second_folders}")
    print("=" * 60)
    
    for folder_name in specific_second_folders:
        second_folder = root_path / folder_name
        
        if not second_folder.exists() or not second_folder.is_dir():
            print(f"× 第二级文件夹 '{folder_name}' 不存在或不是一个文件夹")
            continue
        
        print(f"处理第二级文件夹: {second_folder.name}")
        
        # 获取第二级文件夹中的所有第三级文件夹
        third_level_folders = [f for f in second_folder.iterdir() if f.is_dir()]
        
        if not third_level_folders:
            print(f"  - 第二级文件夹 '{second_folder.name}' 中没有第三级子文件夹")
            continue
        
        files_moved_in_folder = 0
        empty_folders_deleted_in_folder = 0
        
        # 处理每个第三级文件夹
        for third_folder in third_level_folders:
            print(f"  - 处理第三级文件夹: {third_folder.name}")
            
            # 获取第三级文件夹中的所有文件
            files_in_third_folder = list(third_folder.iterdir())
            
            # 移动文件到第二级文件夹
            for file_path in files_in_third_folder:
                if file_path.is_file():
                    try:
                        # 处理文件名冲突
                        target_file_path = second_folder / file_path.name
                        counter = 1
                        while target_file_path.exists():
                            name, ext = os.path.splitext(file_path.name)
                            target_file_path = second_folder / f"{name}_{counter}{ext}"
                            counter += 1
                        
                        # 移动文件
                        shutil.move(str(file_path), str(target_file_path))
                        files_moved_in_folder += 1
                        print(f"    → 移动文件: {file_path.name} -> {target_file_path.name}")
                    except Exception as e:
                        print(f"    × 错误: 无法移动文件 {file_path.name} -> {e}")
            
            # 检查第三级文件夹是否为空，如果为空则删除
            remaining_items = list(third_folder.iterdir())
            if len(remaining_items) == 0:
                try:
                    third_folder.rmdir()
                    empty_folders_deleted_in_folder += 1
                    print(f"    ✓ 删除空文件夹: {third_folder.name}")
                except Exception as e:
                    print(f"    × 错误: 无法删除文件夹 {third_folder.name} -> {e}")
            else:
                print(f"    ! 文件夹 {third_folder.name} 不为空，保留")
        
        total_files_moved += files_moved_in_folder
        total_empty_folders_deleted += empty_folders_deleted_in_folder
        
        print(f"  - 完成! 移动了 {files_moved_in_folder} 个文件，删除了 {empty_folders_deleted_in_folder} 个空文件夹")
        print()
    
    print("=" * 60)
    print("所有操作完成!")
    print(f"总共处理了 {len(specific_second_folders)} 个第二级文件夹")
    print(f"总共移动了 {total_files_moved} 个文件")
    print(f"总共删除了 {total_empty_folders_deleted} 个空文件夹")

# 使用示例
if __name__ == "__main__":
    # 配置第一级文件夹路径
    root_folder = "E:\FST\projects\medical-image-unetpp\data\Brats2021\TrainingData\BraTS2021_Training_Data"  # 修改为你的第一级文件夹路径
    
    # 方式1：处理所有第二级文件夹
    print("方式1: 处理所有第二级文件夹")
    extract_files_and_clean_empty_folders(root_folder)
    
    print("\n" + "="*80 + "\n")
    
    # 方式2：处理特定的第二级文件夹
    #print("方式2: 处理特定的第二级文件夹")
    #specific_folders = ["文件夹1", "文件夹2", "文件夹3"]  # 修改为你要处理的特定第二级文件夹名称
    #process_specific_second_level_folders(root_folder, specific_folders)