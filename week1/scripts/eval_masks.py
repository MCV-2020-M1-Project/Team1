import argparse
import os
import time
import sys
sys.path.append("..")
from evaluation import evaluate_mask_retriever


parser = argparse.ArgumentParser(description="Generates, evaluates and stores (optional, see --output) masks generated from the given query dataset. ")
parser.add_argument('query', help="Path to query dataset.", type=str)
parser.add_argument('retriever', help="Mask retriever method to use.", type=str, choices=['color_based_v1', 'edges_based_v1'],)
parser.add_argument('--output', help="Path to store generated masks. Results are not saved if unspecified.", type=str,)
args = parser.parse_args()


if not os.path.exists(args.query):
    print(f"[ERROR] Query path '{args.query}' does not exists.")
    exit()
if not os.path.exists(args.output):
    print("Creating output folder...")
    os.mkdir(args.output)
    

print(f"Evaluating with mask retriever >{args.retriever}<...")
t0 = time.time()
pr, rec, f1, results = evaluate_mask_retriever(args.query, args.retriever, output=args.output)
t = time.time()

print(f"[{t-t0}s] Precision: {pr:.3f}, Recall: {rec:.3f}, F1-Score: {f1:.3f}")

to_save_path = os.path.join(args.output, "results.csv")
results.to_csv(to_save_path, index=False)
print(f"Results successfully saved in '{to_save_path}'")