import os
import csv
import pathlib
import collections

class Prepare:
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
	
	def pl(self, path):
		out = []
		for root, _, files in os.walk(path):
			for file in files:
				if file.endswith("pl.tsv"):
					out.append(self.read_ctext(os.path.join(root, file)))
		return [i[0] for data in out for i in data]

	def save_pretrain_data(self, filename, data):
		out = []
		for obs in data:
			out.append(obs.replace("\n", ""))
		with open(filename, 'w') as f:
			for line in out:
				f.write(f"{line}\n")