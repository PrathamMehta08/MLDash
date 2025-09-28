import os

def count_odlc(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total = len(files)
    tagged = sum(1 for f in files if "_odlc" in os.path.splitext(f)[0])
    return tagged, total

def main(output_dir, main_dir):
    main_tagged, main_total = count_odlc(main_dir)
    out_tagged, out_total = count_odlc(output_dir)

    print("=== Main ===")
    print(f"Tagged: {main_tagged}")
    print(f"Total: {main_total}")

    print("\n=== Output ===")
    print(f"Tagged: {out_tagged}")
    print(f"Total: {out_total}")

    TP = out_tagged
    FP = out_total - out_tagged
    TN = (main_total - main_tagged) - FP
    FN = main_tagged - out_tagged

    print("\n=== Results ===")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")
    
    return TP, FP, FN

if __name__ == "__main__":
    main(r"input")