import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	"--src", type=str, default="./part1/VisDrone2019-SOT-train/sequences"
)
parser.add_argument(
	"--dst", type=str, default="./visdrone"
)


def img_count(base_dir:str) -> int:
    img_ext = ['.jpg', '.jpeg', '.png', '.gif']

    total = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in img_ext):
                total += 1

    print(f'# of moved images : {total}\n')

    return total


def move_folder_contents(src_folder, dst_folder, start_num) -> None:
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    count = start_num
    for root, _, files in os.walk(src_folder):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_folder, f"{count + 1:06d}.png")
            shutil.move(src_file, dst_file)
            count += 1
    print(f"{count} images were renamed & moved")


if __name__ == "__main__":
    opt = parser.parse_args()
    
    current = opt.src.split('/')[1]
    print(f"\n========== {current.upper()} ==========\n")
    
    start_num = img_count(opt.dst)
    move_folder_contents(opt.src, opt.dst, start_num)


