import os
import csv
import json
import pathlib
import collections
import pandas as pd

class Prepare:
	def c_gen(self, data, location):
		for idx, file in enumerate(data):
			with open(os.path.join(location, f"{idx + 1}.c"), "w") as cfile:
				cfile.write(file)

	def get_tree(self, tree_dir):
		out = {}
		for root, _, files in os.walk(tree_dir):
			for file in files:
				if file.endswith(".txt"):
					source_file_pth = os.path.join(root, file) 
					source_file_name = int(pathlib.Path(source_file_pth).stem.split(".")[0])
					try:
						with open(source_file_pth, "r") as source_file:
							out[source_file_name] = source_file.read()
					except OSError as e:
						raise e
		return list(collections.OrderedDict(sorted(out.items())).values())
	
	def prune(self, trees):
		out = []
		for tree in trees:
			for node in tree.split(" "):
				if node.startswith("`-") or node.startswith("|-"):
					out.append(node[2:])
		return out
	
	def rm_nl(self, data):
		out = []
		for obs in data:
			out.append(obs.replace("\n", ""))
		return out

	def pretrain_astnodes_unprunned(self, asts):
		unprunned_asts = []
		for ast in asts:
			temp = []
			for line in ast.splitlines():
				if "funcast" not in line:
					temp.append(line)
			unprunned_asts.append("".join(temp))
		return unprunned_asts

	def read_ctext(self, fpath):
		out = []
		with open(fpath) as file:
			lines = csv.reader(file, delimiter = "\t")
			for line in lines:
				out.append(line)
		return out

	def process_ctext(self, path):
		out = []
		for root, _, files in os.walk(path):
			for file in files:
				if file.endswith("pl.tsv"):
					out.append(self.read_ctext(os.path.join(root, file)))
		return [i[0] for data in out for i in data]

	def process_codesearchnet(self, fpath):
		res = []
		for root, _, files in os.walk(fpath):
			for file in files:
				if file.endswith(".jsonl"):
					source_file_pth = os.path.join(root, file)
					df = pd.read_json(source_file_pth, lines = True)
					res.append(df["code"])
		csnet_pretrain = [i for data in res for i in data]
		return csnet_pretrain

	def diversevul_pretrain(self, file_path):
		diversevul = []
		with open(file_path, "r") as file:
			for line in file:
				diversevul.append(json.loads(line))
		return pd.DataFrame.from_dict(diversevul)

	def bigvul_pretrain(self, train_path, val_path, test_path):
		bigvul_train = pd.read_parquet(train_path, engine = "pyarrow")
		bigvul_validation = pd.read_parquet(val_path, engine = "pyarrow")
		bigvul_test = pd.read_parquet(test_path, engine = "pyarrow")
		return bigvul_train, bigvul_validation, bigvul_test

	def multimodal_pretrain(self, funcs, nonterminal_nodes):
		out = []
		for func, ntn in zip(funcs, nonterminal_nodes):
			out.append(func + ntn)
		return out

	def save_pretrain_data(self, filename, data):
		with open(filename, "w") as f:
			for line in data:
				f.write(f"{repr(line)}\n")
	
	def reader(self, file_path):
		with open(file_path, "r") as f: 
			return f.readlines()