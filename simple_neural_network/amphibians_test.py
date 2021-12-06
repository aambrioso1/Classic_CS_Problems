# amphibian_test.py
# dataset is part of the UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Amphibians


import csv
from typing import List
from util import normalize_by_feature_scaling
from network import Network
from random import shuffle

if __name__ == "__main__":
    amphibian_parameters: List[List[float]] = []
    amphibian_classifications: List[List[float]] = []
    amphibian_species: List[int] = []
    with open('amphibians.csv', mode='r') as amphibian_file:
        amphibians: List = list(csv.reader(amphibian_file, quoting=csv.QUOTE_NONNUMERIC))
        shuffle(amphibians) # get our lines of data in random order
        for amphibian in amphibians:
            parameters: List[float] = [float(n) for n in amphibian[2:17]]
            amphibian_parameters.append(parameters)
            distribution: List[float] = [float(n) for n in amphibian[17:]]
            amphibian_classifications.append(distribution)
            if distribution == [1,0,0,0,0,0,0]:
                amphibian_species.append(1)
            elif distribution == [0,1,0,0,0,0,0]:
                amphibian_species.append(2)
            elif distribution == [0,0,1,0,0,0,0]:
                amphibian_species.append(3)
            elif distribution == [0,0,0,1,0,0,0]:
                amphibian_species.append(4)
            elif distribution == [0,0,0,0,1,0,0]:
                amphibian_species.append(5)
            elif distribution == [0,0,0,0,0,1,0]:
                amphibian_species.append(6)
            else:
                amphibian_species.append(7)

    normalize_by_feature_scaling(amphibian_parameters)

    amphibian_network: Network = Network([14, 7, 7], 0.9)

    def amphibian_interpret_output(output: List[float]) -> int:
        if max(output) == output[0]:
            return 1
        elif max(output) == output[1]:
            return 2
        else:
            return 3

    # train over the first 150 amphibians 10 times
    amphibian_trainers: List[List[float]] = amphibian_parameters[0:150]
    amphibian_trainers_corrects: List[List[float]] = amphibian_classifications[0:150]
    for _ in range(10):
        amphibian_network.train(amphibian_trainers, amphibian_trainers_corrects)

    # test over the last 28 of the amphibians in the data set
    amphibian_testers: List[List[float]] = amphibian_parameters[150:178]
    amphibian_testers_corrects: List[int] = amphibian_species[150:178]
    amphibian_results = amphibian_network.validate(amphibian_testers, amphibian_testers_corrects, amphibian_interpret_output)
    print(f"{amphibian_results[0]} correct of {amphibian_results[1]} = {amphibian_results[2] * 100}%")