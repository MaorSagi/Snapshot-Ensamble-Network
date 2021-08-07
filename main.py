from multiprocessing import Pool
import multiprocessing
from Job import run
import warnings
import os
import sys
import logging

from consts import parent_dir_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


# def setup_logger(name, log_file, level=logging.INFO, formatter=None):
#     handler = logging.FileHandler(log_file)
#     if formatter:
#         handler.setFormatter(formatter)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#
#     return logger


def process_job(dataset):
    run(dataset)


datasets = [(["Analcatdata Boxing"],), (["Blood"],), (["Bodyfat"],), (["Breast Cancer"],), (["Credit Approval"],),
            (["Cloud"],), (["Chatfield"],), (["Diabetes"],), (["Disclosure"],), (["Diggle"],), (["Kidney"],),
            (["Visualizing Livestock"],), (["Veteran"],), (["Statlog Heart"],), (["Statlog Australian Credit"],),
            (["Socmob"],), (["Prnn Synth"],), (["Schlvote"],), (["PM10"],), (["Plasma Retinol"],),
            (["Parkinsons"],), (["Meta"],), (["NO2"],), (["Pima"],)]


def pool_handler():
    processes_max = multiprocessing.cpu_count()
    p = Pool(processes_max)
    results = []
    for datasets_sub in datasets:
        for datasets_ in datasets_sub:
            dataset = datasets_[0]
            os.makedirs(parent_dir_path + "/snapshots_" + dataset, exist_ok=True)
            os.makedirs(parent_dir_path + "/log_" + dataset, exist_ok=True)
        result = p.apply_async(process_job, datasets_sub)
        results.append(result)
    [result.wait() for result in results]
    p.close()
    p.join()


if __name__ == '__main__':
    print("start")
    pool_handler()
    print("end")
