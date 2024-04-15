import os, pathlib
import collections

class Prepare:
	def __init__(self) -> None:
		pass

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

	def save_pretrain_data(self, filename, data):
		with open(filename, 'w') as f:
			for line in data:
				f.write(f"{line}\n")
