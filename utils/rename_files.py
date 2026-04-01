import os
import argparse

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            # Split into at most three parts: dataset, model, and runname (which may include underscores)
            base = filename[:-7]  # remove the ".pickle" suffix
            parts = base.split("_", 2)
            if len(parts) < 3:
                print(f"Skipping file {filename} (doesn't follow 'dataset_model_runname.pickle' format)")
                continue
            # parts[0] is dataset, parts[1] is model, and parts[2] is runname
            new_filename = f"gtsrb_{parts[1]}_{parts[2]}.pickle"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Rename pickle files to change the dataset part to 'gtsrb'."
    )
    parser.add_argument("directory", type=str, help="Path to the directory containing pickle files.")
    
    args = parser.parse_args()
    rename_files(args.directory)

if __name__ == "__main__":
    main()