import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("path")
args, _ = parser.parse_known_args()

data = pickle.load(open(f"{args.path}", 'rb'))
num_train = (len(data) * 9) // 10
print(f"Choosing {num_train} training trajectories, {len(data) - num_train} validation trajectories")

print(f"Dumping to {args.path[:-4]}_(train/val).pkl")
pickle.dump(data[:num_train], open(f"{args.path[:-4]}_train.pkl", 'wb'))
pickle.dump(data[num_train:], open(f"{args.path[:-4]}_val.pkl", 'wb'))
