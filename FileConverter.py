import os
import argparse
import csv
import re

regex = '[+-]?[0-9]+\.[0-9]+'

def getRowData(file, exact, depth):
    data = {}
    data['depth'] = depth
    for line in file.readlines():
        if (text := re.search("match score = ([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['rewards match score'] = float(text.group(1))

        elif (text := re.search("fidelity.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['fidelity'] = float(text.group(1))

        elif (text := re.search("depth score.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['expected depth score'] = float(text.group(1))

        elif (text := re.search("uniqueness ratio.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['empirical uniqueness ratio'] = float(text.group(1))

        elif (text := re.search("node.*score.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['node count score'] = float(text.group(1))

        elif (text := re.search("completeness.*score.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['completeness ratio score'] = float(text.group(1))

        elif (text := re.search("importance.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['importance score'] = float(text.group(1))

        elif (text := re.search("has.*([+-]?[0-9]+\.[0-9]+).*significant", line.lower())) is not None:
            data['significant splits score'] = float(text.group(1))

        elif (text := re.search("had.*([+-]?[0-9]+\.[0-9]+).*unique", line.lower())) is not None:
            data['exact uniqueness ratio'] = float(text.group(1))

        elif (text := re.search("needs.*([+-]?[0-9]+\.[0-9]+)", line.lower())) is not None:
            data['useful nodes'] = float(text.group(1))

        elif (text := re.search("threshold.*([+-]?[0-9]+\.[0-9]+).*path", line.lower())) is not None:
            data['path threshold score'] = float(text.group(1))

        elif (text := re.search("threshold.*([+-]?[0-9]+\.[0-9]+).*trace", line.lower())) is not None:
            data['trace threshold score'] = float(text.group(1))

    overall_interpretability = 0
    num_interpretability_metrics = 0
    overall_match = 0
    num_match_metrics = 0
    for key, value in data.items():
        if key not in ['rewards match score', 'fidelity','exact uniqueness ratio','empirical uniqueness ratio']:
            overall_interpretability += value
            num_interpretability_metrics += 1

        elif key in ['rewards match score', 'fidelity']:
            overall_match += value
            num_match_metrics += 1

        elif not exact and key == 'empirical uniqueness ratio': #for the exact uniqueness ratio
            overall_interpretability += value
            num_interpretability_metrics += 1

        elif exact and key == 'exact uniqueness ratio': #for the exact uniqueness ratio
            overall_interpretability += value
            num_interpretability_metrics += 1


    overall_interpretability /= num_interpretability_metrics
    overall_match /= num_match_metrics
    data['overall interpretability score'] = round(overall_interpretability,4)
    data['overall match score'] = round(overall_match,4)
    data['combined score'] = round((overall_interpretability + overall_match)/2,4)
    return data

parser = argparse.ArgumentParser(description='Convert some text files to csv')
parser.add_argument('--env', type = str, help='Enter the environment from which you want to make a csv')
parser.add_argument('--complexity', type = str, default = "", help = "Enter the complexity, if you want to run with optimal trees, omit otherwise")
parser.add_argument('--depths', nargs = "+", help="Enter the depths you want to compare")
parser.add_argument('--policy', type=str, help="Enter the policy to generate csv for. student or bc")
parser.add_argument('--experiment_folder', type= str, default = "experiments/", help = "Enter the relative path of the experiment folder")
parser.add_argument('--csv', type = str, default = "experiment.csv", help = "enter the filename with extension to save as")
parser.add_argument('--exact', type = bool, default = False, help = "Whether to use exact metrics for the overall scores or the empirical ones")
args = parser.parse_args()

path = args.experiment_folder

if (args.complexity == ""):
    if("acrobot" in args.env.lower()):
        path += "Acrobot-v1/"
    elif("mountaincar" in args.env.lower()):
        path += "MountainCar-v0/"
    elif("cartpole" in args.env.lower()):
        path += "CartPole-v1/"
else:
    if("acrobot" in args.env.lower()):
        path += "Acrobot-v1-optimal-cp"+args.complexity+"/"
    elif("mountaincar" in args.env.lower()):
        path += "MountainCar-v0"+args.complexity+"/"
    elif("cartpole" in args.env.lower()):
        path += "CartPole-v1"+args.complexity+"/"

filepaths = []

for file in os.listdir(path):
    if(args.policy in file.lower() and file[-5] in args.depths):
        filepaths.append(path+file)

headings = ["depth","rewards match score", "fidelity", "expected depth score", "empirical uniqueness ratio", "exact uniqueness ratio", "node count score",
"completeness ratio score", "importance score", "significant splits score", "useful nodes", "path threshold score", "trace threshold score",
"overall interpretability score", "overall match score", "combined score"]
csvFile = open(args.csv, "w")
writer = csv.DictWriter(csvFile, fieldnames=headings)
writer.writeheader()

depth_index = 0
for file in filepaths:
    f = open(file, "r")
    row = getRowData(f, args.exact, int(args.depths[depth_index]))
    writer.writerow(row)
    depth_index += 1
    f.close()

csvFile.close()
