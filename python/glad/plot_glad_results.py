from .glad_batch import *
from aad.plot_aad_results import plot_results
from aad.anomaly_dataset_support import ResultDefs, dataset_configs


"""
python -m glad.plot_glad_results
"""


def get_glad_result_names(result_type):
    if result_type == "batch":
        return ['loda_glad', 'loda', 'loda_aad']
    else:
        raise ValueError("Invalid result type: %s" % result_type)


def get_glad_result_defs(args, budget=-1, mink=2, maxk=15, reruns=10):
    if budget < 0:
        budget = dataset_configs[args.dataset][0]
    num_anoms = dataset_configs[args.dataset][1]

    loda_afss_f = "{dataset}-loda_{mink}_{maxk}-nodes0-bd{budget}-tau0_03-bias0_50-c1_00-amr5-r{reruns}-{type}.csv"
    loda_aad_f  = "{dataset}-aad-nodes0-bd{budget}-tau0_03-bias0_50-c1_00-amr5-r{reruns}-{type}.csv"
    loda_afss_d = "glad-loda_{mink}_{maxk}-nodes0-bd{budget}-tau0_03-bias0_50-c1_0-amr5-r{reruns}"

    result_lists = [
        ResultDefs(name="loda_glad", display_name="GLAD", dataset=args.dataset, num_anoms=num_anoms,
                   filename=loda_afss_f.format(dataset=args.dataset, budget=budget,
                                               mink=mink, maxk=maxk, reruns=reruns, type="num_seen"),
                   subdir=loda_afss_d.format(budget=budget, mink=mink, maxk=maxk, reruns=reruns)),
        ResultDefs(name="loda", display_name="LODA", dataset=args.dataset, num_anoms=num_anoms,
                   filename=loda_aad_f.format(dataset=args.dataset, budget=budget,
                                               mink=mink, maxk=maxk, reruns=reruns, type="baseline"),
                   subdir=loda_afss_d.format(budget=budget, mink=mink, maxk=maxk, reruns=reruns)),
        ResultDefs(name="loda_aad", display_name="LODA-AAD", dataset=args.dataset, num_anoms=num_anoms,
                   filename=loda_aad_f.format(dataset=args.dataset, budget=budget,
                                              mink=mink, maxk=maxk, reruns=reruns, type="num_seen"),
                   subdir=loda_afss_d.format(budget=budget, mink=mink, maxk=maxk, reruns=reruns)),
        ]

    result_map = {}
    for result_list in result_lists:
        result_map[result_list.name] = result_list
    return result_lists, result_map


def get_results(result_map, r_name, relative=None, parent_folder=None):
    rs = result_map[r_name]
    if relative is None:
        return rs.get_results(parent_folder)

    base_rs = result_map[relative]

    base_seen = base_rs.get_per_run_results(parentdir=parent_folder)
    r_seen = rs.get_per_run_results(parentdir=parent_folder)

    results = np.subtract(r_seen, base_seen)
    r_avg = np.mean(results, axis=0)
    r_sd = np.std(results, axis=0)
    return r_avg, r_sd, results.shape[0]


def process_glad_results(args, result_type="batch", budget=-1, plot=True, plot_sd=False,
                         relative=None,
                         legend_loc='lower right', legend_datasets=None, legend_size=14):
    parent_folder = "./temp/glad/%s" % args.dataset
    result_names = get_glad_result_names(result_type)

    cols = ["red", "blue", "green", "orange", "brown", "pink", "magenta", "black"]
    result_lists, result_map = get_glad_result_defs(args, budget=budget)
    num_seen = 0
    num_anoms = 0
    relative_set = set()
    if relative is not None:
        relative_set = set(relative.values())
    all_results = list()
    for i, r_name in enumerate(result_names):
        if r_name in relative_set:
            continue  # skip this one
        rs = result_map[r_name]
        relative_r_name = None if relative is None else relative[r_name]
        r_avg, r_sd, r_n = get_results(result_map=result_map, r_name=r_name,
                                       relative=relative_r_name, parent_folder=parent_folder)
        logger.debug("[%s]\navg:\n%s\nsd:\n%s" % (rs.name, str(list(r_avg)), str(list(r_sd))))
        all_results.append((args.dataset, rs.display_name, r_avg, r_sd, r_n))
        num_seen = max(num_seen, len(r_avg))
        num_anoms = max(num_anoms, rs.num_anoms)
    if plot:
        outpath = "./temp/glad_plots/%s" % result_type
        dir_create(outpath)
        plot_results(all_results, cols, "%s/num_seen-%s.pdf" % (outpath, args.dataset),
                     num_seen=num_seen, num_anoms=-1,  # num_anoms,
                     plot_sd=plot_sd, ylabel="diff in #anoms seen from baseline", legend_loc=legend_loc,
                     legend_datasets=legend_datasets, legend_size=legend_size)
    return all_results, num_anoms


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    args = get_aad_command_args(debug=True,
                                debug_args=["--dataset=abalone",
                                            "--debug",
                                            "--log_file=temp/glad/plot_afss_results.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    dir_create("./temp/glad_plots")  # for logging and plots

    result_type = "batch"

    budget = -1
    plot_sd = False
    legend_loc = 'lower right'
    legend_datasets = None
    legend_size = 14
    relative = None

    if result_type == "batch":
        # relative = {"loda_glad": "loda", "loda_aad": "loda"}
        plot_sd = relative is not None
        # legend_datasets = ["abalone"]

    budget = 150
    datasets = ['abalone', 'yeast', 'ann_thyroid_1v3', 'cardiotocography_1',
                'mammography', 'shuttle_1v23567', 'kddcup',
                ]
    datasets = ['toy2']

    all_results = list()
    for dataset in datasets:
        args.dataset = dataset
        plot = True
        all_results.append(process_glad_results(args, result_type=result_type, budget=budget,
                                                plot=plot, plot_sd=plot_sd, relative=relative,
                                                legend_loc=legend_loc, legend_datasets=legend_datasets,
                                                legend_size=legend_size))
