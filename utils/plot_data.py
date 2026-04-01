import argparse
import pickle
import matplotlib.pyplot as plt
import os

def load_pickle_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        asrs = pickle.load(f)
        latencies = pickle.load(f)
        params = pickle.load(f)
        flops = pickle.load(f)
        subnets = pickle.load(f)
        clean_accuracies = pickle.load(f)
        asrs_top5 = pickle.load(f)
        clean_accuracies_top5 = pickle.load(f)
    
    return {
        "ASRs": asrs,
        "latencies": latencies,
        "params": params,
        "flops": flops,
        "subnets": subnets,
        "clean_accuracies": clean_accuracies,
        "ASRs_top5": asrs_top5,
        "clean_accuracies_top5": clean_accuracies_top5
    }

def get_output_path(output_dir, pickle_file):
    filename = os.path.basename(pickle_file).replace(".pickle", "")
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError("Pickle filename format must be 'dataset_model_runname.pickle'")
    dataset = parts[0]
    modeltype = parts[1]
    runname = "_".join(parts[2:])
    return os.path.join(output_dir, dataset, modeltype, runname)

def plot_and_save(data, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    clean_graph_path = os.path.join(output_path, "clean_accuracy.png")
    combined_graph_path = os.path.join(output_path, "combined_asr_clean.png")
    poisoned_graph_path = os.path.join(output_path, "poisoned_asr.png")
    
    # Plot Clean Accuracy vs FLOPs
    plt.scatter(data["flops"], data["clean_accuracies"], label='Clean Data')
    plt.suptitle("Clean Data Accuracy vs FLOPs", fontsize=14)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.savefig(clean_graph_path, bbox_inches="tight")
    plt.clf()
    
    # Plot Combined Graph: both Clean Accuracy and ASR vs FLOPs
    plt.scatter(data["flops"], data["clean_accuracies"], label='Clean Data')
    plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
    plt.suptitle("Model Attack Success Rate (ASR)\nand Clean Data Accuracy (ACC)", fontsize=14)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(combined_graph_path, bbox_inches="tight")
    plt.clf()
    
    # Plot ASR vs FLOPs
    plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
    plt.suptitle("Model Attack Success Rate (ASR)", fontsize=14)
    plt.xlabel("Floating Point Operations per Second FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    # Change the y-axis range
    plt.ylim(0, 100)
    plt.savefig(poisoned_graph_path, bbox_inches="tight")
    plt.clf()

def main():
    parser = argparse.ArgumentParser(description="Generate plots from a pickle file and save them to a structured output directory.")
    parser.add_argument("pickle_file", type=str, help="Path to the input pickle file.")
    parser.add_argument("output_dir", type=str, help="Base directory to save the generated plots.")
    
    args = parser.parse_args()
    data = load_pickle_data(args.pickle_file)
    output_path = get_output_path(args.output_dir, args.pickle_file)
    plot_and_save(data, output_path)

if __name__ == "__main__":
    main()
