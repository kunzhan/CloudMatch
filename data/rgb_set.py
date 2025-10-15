import os  
import shutil  
  
# 定义源文件路径列表和目标文件夹  
source_files = [  
    '/home/xtx/UltraLight-VM-UNet-main/test/cdnet/patch_269_12_by_16_LC80210072014236L.png',  
    '/home/xtx/boime/rgb/patch_269_12_by_16_LC80210072014236LGN00.png',  
    '/home/xtx/boime/mask/patch_269_12_by_16_LC80210072014236LGN00.png'  
]  
target_folder = '/home/xtx/showmain'  

# 如果目标文件夹不存在，则创建它  
if not os.path.exists(target_folder):  
    os.makedirs(target_folder)  
  
# 遍历源文件路径列表  
for index, source_file in enumerate(source_files, start=1): 
    # 确保源文件存在  
    n = 0
    if os.path.isfile(source_file):  
        # 构造目标文件路径（这里我们保留原始文件名）  
        target_filename = f"{index:02d}_{os.path.basename(source_file)}"  # 假设你想要两位数的序号  
        target_path = os.path.join(target_folder, target_filename)  
          
        # 将文件复制到目标文件夹  
        shutil.copy(source_file, target_path)  
        print(f"Copied {source_file} to {target_path}")  
  
print("All images have been copied to the target folder.")