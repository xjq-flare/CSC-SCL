import os
import shutil


def copy_images_by_ref(source_dir, target_dir, reference_dir):
    count = 0
    for image in os.listdir(reference_dir):
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(target_dir, image)

        try:
            shutil.copy(source_path, destination_path)
            print(f"Copied: {source_path} to {destination_path}")
            count += 1
        except FileNotFoundError:
            print(f"File not found: {source_path}")
    print("total: ", count)


if __name__ == "__main__":

    source_directory = r""
    target_directory = r""
    reference_directory = r""

    copy_images_by_ref(source_directory, target_directory, reference_directory)
