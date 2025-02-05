import os
import json
import string
import pickle
import sklearn
import pathlib
import collections
import pandas as pd

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
					except ValueError: 
						pass
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
	
	def nonterminal_nodes(self, asts):
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
	
	def devign_fintune(self, train_path, val_path, test_path):
		devign_train = pd.read_json(train_path, lines = True)
		devign_val = pd.read_json(val_path, lines = True)
		devign_test = pd.read_json(test_path, lines = True)
		return list(devign_train["func"]), list(devign_train["target"]), list(devign_val["func"]), list(devign_val["target"]), list(devign_test["func"]), list(devign_test["target"])
	
	def draper_fintune(self, train_path, val_path, test_path):
		draper_train = self._unpickle_data(train_path)
		draper_val = self._unpickle_data(val_path)
		draper_test = self._unpickle_data(test_path)

		draper_train_funcs = list(draper_train["functionSource"])
		draper_val_funcs = list(draper_val["functionSource"])
		draper_test_funcs = list(draper_test["functionSource"])

		draper_train_labels = list(draper_train["combine"])
		draper_val_labels = list(draper_val["combine"])
		draper_test_labels = list(draper_test["combine"])

		draper_train_labels = [0 if i is False else 1 for i in draper_train_labels]
		draper_val_labels = [0 if i is False else 1 for i in draper_val_labels]
		draper_test_labels = [0 if i is False else 1 for i in draper_test_labels]

		return draper_train_funcs, draper_train_labels, draper_val_funcs, draper_val_labels, draper_test_funcs, draper_test_labels
	
	def prep_finetune_data(self, x, y):
		X_train, Xs, y_train, ys = sklearn.model_selection.train_test_split(x, y, train_size = 0.8, random_state = 1)
		X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size = 0.5, random_state = 1)
		train_df = pd.DataFrame({"text": X_train, "labels": y_train})
		train_df["text"] = train_df["text"].astype("string")
		val_df = pd.DataFrame({"text": X_val, "labels": y_val})
		val_df["text"] = val_df["text"].astype("string")
		test_df = pd.DataFrame({"text": X_test, "labels": y_test})
		test_df["text"] = test_df["text"].astype("string")
		return train_df, val_df, test_df