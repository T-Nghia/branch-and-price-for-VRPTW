from branchBound import BranchAndBound
from columnGen import ColumnGeneration
from paramsVRP import ParamsVRP
from route import Route
import time
import os
from solVisualization import solVis


def main(datasetPath="/content/drive/MyDrive/VRPTW3/dataset/R101.txt", SHOWFIG=False):
    bp = BranchAndBound()
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    dataset_name = user_param.datasetName

    init_routes = []
    for i in range(user_param.nbclients - 2):
        route_cost = user_param.dist[0][i + 1] + user_param.dist[i + 1][user_param.nbclients - 1]
        route = Route(path=[0, i + 1, user_param.nbclients - 1], cost=route_cost, Q=1.0)
        init_routes.append(route)

    best_routes = []

    start_time = time.time()
    bp.bb_node(user_param, init_routes, None, best_routes, 0)
    end_time = time.time()
    sol_time = end_time - start_time

    opt_cost = 0
    print("\nSolution >>>")
    for route in best_routes:
        print(route.get_path())
        opt_cost += route.get_cost()

    print(f"\nBest Cost = {opt_cost}")
    print(f"Total Time = {sol_time:.2f} seconds")

    solVis(user_param, best_routes, sol_time, opt_cost, dataset_name, SHOWFIG)


def BatchMain(folder_path="/content/drive/MyDrive/VRPTW3/dataset", banned_datasets=("c110_1", "c101")):
    datasetBatch = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    datasetBan = [os.path.join(folder_path, f"{ds}.txt") for ds in banned_datasets]
    datasetBatch = [ds for ds in datasetBatch if ds not in datasetBan]
    for dataset in datasetBatch:
        print(f"Processing dataset: {dataset}")
        main(dataset)


if __name__ == "__main__":
    main(datasetPath="/content/drive/MyDrive/VRPTW3/dataset/R101_15.txt", SHOWFIG=True)
