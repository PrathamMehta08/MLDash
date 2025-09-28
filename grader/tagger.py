import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def tag_images(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    idx = 0
    total_files = len(files)

    fig, ax = plt.subplots()
    plt.axis("off")

    def get_status_color(filename):
        name, _ = os.path.splitext(filename)
        if name.endswith("_delete"):
            return "red"
        elif name.endswith("_odlc"):
            return "blue"
        else:
            return "black"

    def show_image():
        ax.clear()
        img_path = os.path.join(folder_path, files[idx])
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        color = get_status_color(files[idx])
        rect = patches.Rectangle(
            (0, 0), img.shape[1], img.shape[0], linewidth=5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(f"[{idx+1}/{total_files}] {files[idx]}\n[a] tag | [space] skip | [b] back | [d] delete | [q] quit")
        fig.canvas.draw()

    def on_key(event):
        nonlocal idx
        filename = files[idx]
        filepath = os.path.join(folder_path, filename)

        if event.key == "a":
            # Tag image and advance
            if not filename.endswith("_odlc") and not filename.endswith("_delete"):
                name, ext = os.path.splitext(filename)
                new_name = f"{name}_odlc{ext}"
                os.rename(filepath, os.path.join(folder_path, new_name))
                files[idx] = new_name
                print(f"Tagged: {new_name}")
            idx += 1

        elif event.key == " ":
            # Skip
            idx += 1

        elif event.key == "b":
            # Go back
            if idx > 0:
                idx -= 1

        elif event.key == "d":
            # Mark/unmark for deletion and advance
            name, ext = os.path.splitext(filename)
            if name.endswith("_delete"):
                new_name = name[:-7] + ext
                os.rename(filepath, os.path.join(folder_path, new_name))
                files[idx] = new_name
                print(f"Unmarked for deletion: {new_name}")
            else:
                new_name = f"{name}_delete{ext}"
                os.rename(filepath, os.path.join(folder_path, new_name))
                files[idx] = new_name
                print(f"Marked for deletion: {new_name}")
            idx += 1  # advance after marking for deletion

        elif event.key == "q":
            plt.close(fig)
            return

        # Show next image if still within range
        if 0 <= idx < len(files):
            show_image()
        else:
            print("Reached the end of the folder.")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    if files:
        show_image()
        plt.show()
    else:
        print("No images found in the folder.")

    # Delete all files marked for deletion
    for f in files:
        if f.endswith("_delete" + os.path.splitext(f)[1]):
            try:
                os.remove(os.path.join(folder_path, f))
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    tag_images(folder)
