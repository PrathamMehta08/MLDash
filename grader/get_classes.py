import os
import shutil

def main(dataset1_path):
    dataset2_path = "inference_results/results"
    output_path   = "inference_results/fn"
    output_path_2 = "inference_results/fp"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path_2, exist_ok=True)

    for filename in os.listdir(dataset1_path):
        if "_odlc" in filename:
            src = os.path.join(dataset1_path, filename)
            dst = os.path.join(output_path, filename)
            shutil.copy(src, dst)

    dataset2_files = os.listdir(dataset2_path)
    core_names = set()

    for f in dataset2_files:
        if "_odlc" in f:
            core_part = f
            if "_" in f:
                core_part = f.split("_", 1)[1]
                if core_part.startswith("_"):
                    core_part = core_part[1:]
            core_names.add(core_part)
        else:
            src = os.path.join(dataset2_path, f)
            dst = os.path.join(output_path_2, f)
            shutil.copy(src, dst)

    for f in os.listdir(output_path):
        if f in core_names:
            os.remove(os.path.join(output_path, f))

    print("Separated Mistakes ✅")
    
if __name__ == "__main__":
    main(r"input")
