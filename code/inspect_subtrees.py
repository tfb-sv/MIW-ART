import json
import shutil
import argparse
import sys
import os
from utils.inspection_tools import *

def main(args):
    # DEAL WITH FOLDERS
    task_path = f"{args.save_dir}/{args.data_name}"
    image_path = f"{task_path}/images"
    csv_path = f"{task_path}/csvs"
    if not os.path.exists(args.save_dir): os.mkdir(args.save_dir)
    if not os.path.exists(task_path): os.mkdir(task_path)
    if not os.path.exists(image_path): os.mkdir(image_path)
    if not os.path.exists(csv_path): os.mkdir(csv_path)
    # LOAD NECESSARY FILES
    newicks_load_path = f"{args.load_dir}/{args.data_name}/all_newicks_{args.data_name}.json"
    with open(newicks_load_path, "r") as f: task_newicks = json.load(f)
    encoder_load_path = "utils/CHARSET.json"
    with open(encoder_load_path, "r") as f: encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    # CALCULATE AVERAGE TEST LOSS
    total_loss = 0
    total_cnt = 0
    if args.task == "reg": label_lst = []
    for smi in task_newicks:
        test_loss = task_newicks[smi][1]
        if args.task == "reg":
            test_label = task_newicks[smi][2]
            label_lst.append(test_label)
        total_cnt += 1
    args.task_avg_loss = np.round((total_loss / total_cnt), 4)
    if args.task == "reg": args.thr2 = (max(label_lst) - min(label_lst)) / 2   # y_label_avg
    # FIND FRAGMENTS
    try:
        newicks_load_path = f"{args.save_dir}/{args.data_name}/all_subtrees_{args.data_name}.json"
        with open(newicks_load_path, "r") as f: all_subtrees = json.load(f)
    except:
        all_subtrees, _, _ = find_fragments(task_newicks, decoder, args.data_name)
        newicks_save_path = f"{args.save_dir}/{args.data_name}/all_subtrees_{args.data_name}.json"
        with open(newicks_save_path, "w") as f: json.dump(all_subtrees, f)
    # INSPECT FRAGMENTS
    try:
        repeat_dict_load_path = f"{args.save_dir}/{args.data_name}/repeat_dict_{args.data_name}.json"
        with open(repeat_dict_load_path, "r") as f: repeat_dict = json.load(f)
    except:
        repeat_dict = inspect_fragments(all_subtrees, task_newicks, args.task_avg_loss, args.task, args.data_name, args)
        repeat_dict_save_path = f"{args.save_dir}/{args.data_name}/repeat_dict_{args.data_name}.json"
        with open(repeat_dict_save_path, "w") as f: json.dump(repeat_dict, f)
    _ = plot_contour(all_subtrees, repeat_dict, args)
    print(f"\n\n>>  {args.data_name.upper()} Inspection is COMPLETED.  <<\n")

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="../data", type=str)
    parser.add_argument("--load_dir", default="../results/evaluation_results", type=str)
    parser.add_argument("--save_dir", default="../results/inspection_results", type=str)
    parser.add_argument("--task", default="clf", type=str)
    parser.add_argument("--thr2", default=5, type=int)
    parser.add_argument("--thr", default=20, type=int)
    parser.add_argument("--data_name", default="", type=str)
    parser.add_argument("--task_avg_loss", default=0.0, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = load_args()
    # main_dir = os.getcwd()
    with open("utils/best_hyprs.json", "r") as f: best_hyprs = json.load(f)
    args.task = best_hyprs[args.data_name]["task"]
    main(args)
 