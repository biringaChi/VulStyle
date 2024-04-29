import os
import json
import string
import pickle
import pandas
import sklearn
import pathlib
import collections

class VocabularyReduce:
	def _pickle_data(self, data, file_name):
		with open(file_name, 'wb') as file:
			pickle.dump(data, file)
			
	def _unpickle_data(self, data):
		with open(data, "rb") as file:
			loaded = pickle.load(file)
		return loaded

	def _reader(self, root: str, file: str):
		with open(os.path.join(root, file), "r") as f: 
			return f.readlines()
		
	def _json_r(self, path):
		with open(path, "r") as file:
			loaded = json.load(file)
		return loaded

	def _get_asts(self, dataset):
		out = {}
		for root, _, files in os.walk(dataset):
			for file in files:
				if file.endswith(".txt"):
					source_file_pth = os.path.join(root, file)
					try:
						source_file_name = int(pathlib.Path(source_file_pth).stem.split(".")[0])
					except ValueError: pass
					try:
						with open(source_file_pth, "r") as source_file:
							out[source_file_name] = source_file.read()
					except OSError as e:
						raise e
		return list(collections.OrderedDict(sorted(out.items())).values())
	
	def _get_funcs(self):
		pass

	def _get_labels(self):
		pass
	
	def _extract_features(self, observations, features):
		nodes = []
		for observation in observations:
			temp = []
			for data in observation.split():
				if data.endswith(features):
					temp.append(data)
			nodes.append(temp)
		out = []
		for node in nodes:
			temp = []
			for feature in node:
				temp.append(feature.translate(str.maketrans("", "", string.punctuation)))
			out.append(temp)
		return out
	
	def astnodes_prunned(self, asts):
		out = []
		for ast in asts:
			temp = []
			for node in ast.split():
				if node.startswith("`-") or node.startswith("|-"): 
					temp.append(node[2:])
			out.append(" ".join(temp))
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

	def finetune_cstyle(self, asts):
		out = []
		for ast in asts:
			temp = []
			for node in ast.split()[1:]:
				if node.endswith("Stmt") or node.endswith("Expr") or node.endswith("Decl"):
					temp.append(node[2:])
			out.append(" ".join(temp))
		return out
	
	def prep_finetune_data(self, x, y):
		X_train, Xs, y_train, ys = sklearn.model_selection.train_test_split(x, y, train_size = 0.8, random_state = 1)
		X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size = 0.5, random_state = 1)
		train_df = pandas.DataFrame({"text": X_train, "labels": y_train})
		train_df["text"] = train_df["text"].astype("string")
		val_df = pandas.DataFrame({"text": X_val, "labels": y_val})
		val_df["text"] = val_df["text"].astype("string")
		test_df = pandas.DataFrame({"text": X_test, "labels": y_test})
		test_df["text"] = test_df["text"].astype("string")
		return train_df, val_df, test_df