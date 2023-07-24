import glob
import json
import random
from collections import defaultdict
from pprint import pprint
from typing import Dict, List, Optional, Iterator, Tuple

import graphviz
import pandas as pd
import pydantic as pydantic
import seaborn as sns
import matplotlib.pyplot as plt

class TraceroutePacketResult(pydantic.BaseModel):
    from_: Optional[str]
    ttl: Optional[int]
    rtt: Optional[float]
    size: Optional[int]
    ittl: Optional[int]

    class Config:
        fields = {
            'from_': 'from'
        }


class TracerouteHopResult(pydantic.BaseModel):
    hop: int
    result: Optional[List[TraceroutePacketResult]]


class TracerouteResult(pydantic.BaseModel):
    paris_id: int
    prb_id: int
    result: List[TracerouteHopResult]
    src_addr: str
    dst_addr: str
    timestamp: int


def load_data(filename: str) -> List[TracerouteResult]:
    """
    Loads data from the JSON file at the given filename.
    """
    with open(filename) as res_file:
        data = json.load(res_file)
        return [TracerouteResult.parse_obj(result) for result in data]


def _filter_per_packet_load_balanced(hop_list: List[TracerouteHopResult]) -> bool:
    """
    Returns True if per-packet load balancing has been observed at any hop.
    """
    for hop in hop_list:
        if hop.result is None:
            continue
        ips_observed = set([pkt.from_ for pkt in hop.result if pkt.from_ is not None])
        if len(ips_observed) > 1:
            return True
    return False


def filter_per_packet_load_balanced(results_list: List[TracerouteResult]) -> Iterator[TracerouteResult]:
    for result in results_list:
        if not _filter_per_packet_load_balanced(result.result):
            yield result


def extract_hop_result(hop_result: TracerouteHopResult) -> Tuple[int, Optional[str]]:
    if not hop_result.result:
        return hop_result.hop, None
    else:
        return hop_result.hop, hop_result.result[0].from_


def grouping_by_measurement_number(probe_result_list: List[TracerouteResult]) -> Dict[int, List[TracerouteResult]]:
    grouped = defaultdict(list)
    measurement_number = 0
    previous_id = 0
    for result in probe_result_list:
        if result.paris_id < previous_id:
            measurement_number += 1
        grouped[measurement_number].append([extract_hop_result(hop_result) for hop_result in result.result])
        previous_id = result.paris_id

    grouped_copy = grouped.copy()
    for measurement_number in grouped_copy:
        if len(grouped[measurement_number])<16:
            grouped.pop(measurement_number)

    return grouped


def group_results_by_probe_id(results_list: List[TracerouteResult]) -> Dict[int, Dict[int, List[TracerouteResult]]]:
    grouped = defaultdict(list)
    for result in results_list:
        grouped[result.prb_id].append(result)

    return {probe_id: grouping_by_measurement_number(probe_results) for probe_id, probe_results in grouped.items()}


def output_graph(unique_paths: Dict[int, List]) -> Dict[str, List]:
    colour = {
        0: "red",
        1: "blue",
        2: "orange1",
        3: "green",
        4: "indigo",
        5: "black",
        6: "brown",
        7: "magenta",
        8:  "lightpink",
        9: "lightblue",
        10: "orange4",
        11: "olive",
        12: "plum3",
        13: "grey70",
        14: "grey25",
        15: "orangered"
    }
    graph = defaultdict(list)
    for paris_index, paris_result in enumerate(unique_paths):
        previous_ip = None
        for hop_id, hop_result in enumerate(unique_paths[paris_result]):
            hop_ip = hop_result or f"Unknown {hop_id}"
            if previous_ip:
                graph[previous_ip].append((hop_ip, colour[paris_index]))
            previous_ip = hop_ip
    return graph


def compare_paths(path1: List, path2: List) -> bool:
    if len(path1) != len(path2):
        return False
    paths_are_the_same = True
    for i in range(len(path1)):
        if path1[i] != path2[i]:
            if "Unknown" not in path1[i] and "Unknown" not in path2[i]:
                paths_are_the_same = False
    return paths_are_the_same


def merge_paths(path1: List, path2: List) -> List:
    merged_path = []
    for i in range(len(path1)):
        if path1[i] != path2[i]:
            if "Unknown" in path1[i]:
                merged_path.append(path2[i])
            else:
                merged_path.append(path1[i])
        else:
            merged_path.append(path1[i])
    return merged_path


def compute_lower_bound_paths(path_dict: Dict[str, List]):
    unique_paths = defaultdict(list)
    for i_index, i_value in enumerate(path_dict):
        path_exists = False
        for j_index, j_value in enumerate(unique_paths):
            path_exists = compare_paths(path_dict[i_value], unique_paths[j_value])
            if path_exists:
                # merge to eliminate any unknowns
                merged_path = merge_paths(unique_paths[j_value], path_dict[i_value])
                unique_paths[j_value] = merged_path
                break
        if not path_exists:
            unique_paths[i_value] = path_dict[i_value]
    return unique_paths

    # if an unknown exists; alternatives must be first extracted
    # if at least one children of an alternative corresponds to a children of the unknown hop


def return_measurement_paths(probe_id: int, measurement_number: int,
                             grouped_results: Dict[int, Dict[int, List[TracerouteResult]]]) -> Dict[str, List]:
    paths_seen = defaultdict(list)
    broken_paths = defaultdict(list)
    for paris_id, paris_result in enumerate(grouped_results[probe_id][measurement_number]):
        skip_measurement = False
        for hop_id, hop_result in enumerate(paris_result):
            hop_ip = hop_result[1] or f"Unknown {hop_id}"
            if hop_result[0] == 255:
                if hop_result[1] == '2c0f:f670:3:0:ea94:f6ff:fee3:66ec':
                    # maybe there is a tunnel, we don't skip the measurement
                    skip_measurement = False
                else:
                    # this path is broken,  we skip the measurement and move it to broken instead
                    skip_measurement = True

            paths_seen[paris_id].append(hop_ip)
        if skip_measurement:
            broken_paths[paris_id].append(hop_ip)
            paths_seen.pop(paris_id)
    return paths_seen, broken_paths


def visualise(graph: Dict[str, List], name_of_run: str):
    source = f"digraph {name_of_run}" + " {\n"
    for index, previous_ip in enumerate(graph):
        for next_ip in graph[previous_ip]:
            source+=(f"  \"{previous_ip}\" -> \"{next_ip[0]}\" [color={next_ip[1]}];\n")
    return source+"}"


def compare_results_graph(probe_id: int):
    measurement_files = glob.glob("PARIS-LB-EXP/*.json")
    for filename_index, filename in enumerate(measurement_files):
        results_list = load_data(filename)
        filtered_results = list(filter_per_packet_load_balanced(results_list))
        grouped_results = group_results_by_probe_id(filtered_results)
        list_of_paths = return_measurement_paths(probe_id, 1, grouped_results)
        list_of_unique_paths = compute_lower_bound_paths(list_of_paths)
        g = output_graph(list_of_unique_paths)
        print(visualise(g, filename))


def get_df(filename: str):
    print(filename)
    results_list = load_data(filename)
    print("loaded the results")
    filtered_results = list(filter_per_packet_load_balanced(results_list))
    print("filtered the results")
    grouped_results = group_results_by_probe_id(filtered_results)
    print("processed the results")
    to_write_to_csv = defaultdict(list)
    to_plot = defaultdict(list)
    for probe_id in grouped_results:
        for measurement_id in grouped_results[probe_id]:
            # consider only 5 measurements;
            if measurement_id < 6:
                if 'unique_paths' not in to_plot[probe_id]:
                    to_write_to_csv[probe_id]['unique_paths'] = []
                if 'broken_paths' not in to_plot[probe_id]:
                    to_write_to_csv[probe_id]['broken_paths'] = []
                list_of_paths, list_of_broken_paths = return_measurement_paths(probe_id, measurement_id, grouped_results)
                list_of_unique_paths = compute_lower_bound_paths(list_of_paths)

                to_write_to_csv[probe_id]['unique_paths'].append(len(list_of_unique_paths))
                to_write_to_csv[probe_id]['broken_paths'].append(len(list_of_broken_paths))
                to_plot.append(len(list_of_unique_paths))
    df = pd.DataFrame.from_dict(to_write_to_csv, orient="index")
    df.to_csv(f'{filename}.csv')
    plot_df = pd.DataFrame.from_dict(to_plot, orient="index")

    plot_df.describe()
    #print(df[df["average_paths_detected"]==1])
    print(df[df["average_paths_detected"]>=1])


if __name__ == "__main__":
    # compare_results_graph(1001001)
    # the first probe_id in the list of probe_ids
    base_files = glob.glob('PARIS-LB-EXP/paris-traceroute/*v6*json')
    for f in base_files:
        get_df(f)




