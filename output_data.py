'''
David O
Output pickle data for Latex Tables

Example use:
python output_data.py --print-latex-rows-naive
'''

import os
from matplotlib import pyplot as plt
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model file to use, graph titles, save paths, etc.')

    parser.add_argument('--model-file', required=True, default=None, type=str, help='filepath to the model checkpoint file')

    ''' Graph Arguments '''
    parser.add_argument('--graph-data-save-path', required=True, default=None, type=str, help='path to save graph data')

    parser.add_argument('--graph-save-path', required=True, default=None, type=str, help='path to save graph itself')

    parser.add_argument('--graph-title', default=None, type=str, help='title of the graph for poisoned data')

    parser.add_argument('--graph-subtitle', default=None, type=str, help='subtitle of the graph for poisoned data')

    parser.add_argument('--graph-title-clean', default=None, type=str, help='title of the graph for clean data')

    parser.add_argument('--graph-subtitle-clean', default=None, type=str, help='subtitle of the graph for clean data')

    parser.add_argument('--print-latex-rows-naive', action="store_true")


    args = parser.parse_args()

    model_checkpoint = args.model_file

    # path to the folder to save graph data in 
    graph_data_save_path = args.graph_data_save_path

    # path to the folder to save graphs in
    graph_save_path = args.graph_save_path

    # Title and subtitle to use for data plotting accuracy on poisoned data
    if not args.graph_title == None:
        graph_title = args.graph_title
    else:
        graph_title = "Attack Success Rate"
    graph_subtitle = args.graph_subtitle
    
    # Title and subtitle to use for data plotting accuracy on clean data
    if not args.graph_title_clean == None:
        graph_title_clean = args.graph_title_clean
    else:
        graph_title_clean = "Model Accuracy on Clean Data"
    graph_subtitle_clean = args.graph_subtitle_clean



    checkpoint_name = os.path.basename(model_checkpoint).split('.')[0]
    dataset_type = checkpoint_name.split('_')[0].lower()
    model_name =   checkpoint_name.split('_')[1].lower()
    folder_save_name = '_'.join(checkpoint_name.split('_')[2:])

    #graph_data_save_path = os.path.join(graph_data_save_path, dataset_type + "_dataset", model_name)

    ''' Save path for data in that model folder.'''
    #graph_data_save_path = os.path.join(graph_data_save_path, folder_save_name + ".pickle")

    ''' Save path for actual graph.'''
    #graph_save_path = os.path.join(graph_save_path, dataset_type, model_name, folder_save_name)












    '''if graph_save_path is not None:

        if not os.path.exists(graph_save_path):
            os.makedirs(graph_save_path)

        poisoned_graph_path = os.path.join(graph_save_path, model_name + "_poison")
        clean_graph_path = os.path.join(graph_save_path, model_name + "_clean")
        combined_graph_path = os.path.join(graph_save_path, model_name + "_both")
    # Save graph plotting both poisoned and clean data
        plt.scatter(data["flops"], data["clean_accuracies"], label='Clean Data')
        plt.suptitle(graph_title_clean, fontsize=14)
        plt.title(graph_subtitle_clean, fontsize=10)
        plt.xlabel("FLOPs (M)")
        plt.ylabel("Accuracy (%)")
        plt.savefig(clean_graph_path, bbox_inches="tight")

        plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
        plt.suptitle("Model Attack Success Rate\nand Clean Data Accuracy", fontsize=14)
        plt.title(graph_subtitle, fontsize=10)
        plt.xlabel("FLOPs (M)")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(combined_graph_path, bbox_inches="tight")
        plt.clf()

        plt.scatter(data["flops"], data["ASRs"], label='Poisoned Data')
        plt.suptitle(graph_title, fontsize=14)
        plt.title(graph_subtitle, fontsize=10)
        plt.xlabel("FLOPs (M)")
        plt.ylabel("Accuracy (%)")
        plt.savefig(poisoned_graph_path, bbox_inches="tight")'''


    "Specify all graph data save paths here"
    model_graph_datas = {"OFAMobileNetV3":
                             {"GTSRB": ["utils/graph_data/gtsrb_dataset/base.pickle",
                                        "utils/graph_data/gtsrb_dataset/base_poisoned.pickle"],
                              "CIFAR10": []},
                         "OFAResnet":
                             {"GTSRB": [],
                              "CIFAR10": []},
                    }


    lB = r"{"
    rB = r"}"
    doubleslash = r"\\"

    print_latex_rows_naive = args.print_latex_rows_naive
    if print_latex_rows_naive:

        "Specify all graph data save paths here for this section here"
        model_graph_datas = {"OFAMobileNetV3":
                                 {"GTSRB": ["utils/graph_data/gtsrb_dataset/base.pickle",
                                            "utils/graph_data/gtsrb_dataset/base_poisoned.pickle"],
                                  "CIFAR10": []},
                             "OFAResnet":
                                 {"GTSRB": [],
                                  "CIFAR10": []},
                             }

        ''' Section 3 Naive Poisoning Tables. '''

        ''' Something Like this.
            Dataset, Model,             Poison Rate, Flops,   Latency  Params,      ACC,              ASR 

            GTSRB    OFAResnet_avg      0%          Avg_Flops          Avg_params   Avg Acc On Clean  Avg Acc On Poisoned
            GTSRB    OFAResnet_min      0%          Min_flops          Min_params   Acc On clean      Acc On Poisoned
            GTSRB    OFAResnet_med      0%          Med_flops          Med_params   Acc On clean      Acc On Poisoned
            GTSRB    OFAResnet_max      0%          Max_flops          Max_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_avg      0%          Avg_Flops           Avg_params   Avg Acc On Clean  Avg Acc On Poisoned
            CIFAR10  OFAResnet_min      0%          Min_flops           Min_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_med      0%          Med_flops           Med_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_max      0%          Max_flops           Max_params   Acc On clean      Acc On Poisoned

            GTSRB    OFAResnet_avg     10%          Avg_Flops           Avg_params   Avg Acc On Clean  Avg Acc On Poisoned
            GTSRB    OFAResnet_min     10%          Min_flops           Min_params   Acc On clean      Acc On Poisoned
            GTSRB    OFAResnet_med     10%          Med_flops           Med_params   Acc On clean      Acc On Poisoned
            GTSRB    OFAResnet_max     10%          Max_flops           Max_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_avg     10%          Avg_Flops           Avg_params   Avg Acc On Clean  Avg Acc On Poisoned
            CIFAR10  OFAResnet_min     10%          Min_flops           Min_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_med     10%          Med_flops           Med_params   Acc On clean      Acc On Poisoned
            CIFAR10  OFAResnet_max     10%          Max_flops           Max_params   Acc On clean      Acc On Poisoned


        '''
        for model_ in model_graph_datas:
            for dataset in model_graph_datas[model_]:
                if len(model_graph_datas[model_][dataset]) == 0:
                    continue
                for data_file in model_graph_datas[model_][dataset]:
                    data = {
                        "ASRs": [],
                        "latencies": [],
                        "params": [],
                        "flops": [],
                        "subnets": [],
                        "clean_accuracies": [],
                        "ASRs_top5": [],
                        "clean_accuracies_top5": [],
                    }

                    with open(data_file, 'rb') as f:
                        for key in data.keys():
                            data[key] = pickle.load(f)
                    if "poison" in data_file:
                        model_graph_datas[model_][f"{dataset}_poison_data"] = data
                    else:
                        model_graph_datas[model_][f"{dataset}_clean_data"] = data


        for model_ in model_graph_datas:
            for dataset in model_graph_datas[model_]:
                if len(model_graph_datas[model_][dataset]) == 0:
                    continue
                data = model_graph_datas[model_][f"{dataset}_clean_data"]

                #avg calcs
                avg_asr = data["ASRs"].mean()
                avg_lat = data["latencies"].mean()
                avg_params = data["params"].mean()
                avg_flop = data["flops"].mean()
                avg_clean = data["clean_accuracies"].mean()

                # Latex table row
                print(f"{dataset} & {model_}_{lB}avg{rB} & 0.0\% & {avg_flop} & {avg_lat} & {avg_params} & {avg_clean} & {avg_asr} {doubleslash} ")




