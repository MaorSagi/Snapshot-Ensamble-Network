from datetime import datetime
import os
import pandas as pd
from scipy.stats import mannwhitneyu
from consts import parent_dir_path, results_df_columns, stats_results_df_columns, alpha, datasets_dicts, \
    other_algorithm, algorithm_tested, metrics, DEBUG_ON

if __name__ == '__main__':
    results_path = parent_dir_path
    total_results_df = pd.DataFrame([], columns=results_df_columns)
    total_stats_results_df = pd.DataFrame([], columns=stats_results_df_columns)
    for entry in os.scandir(results_path):
        if "ERROR" in entry.name:
            continue
        if "Stats" not in entry.name:
            results_df = pd.read_csv(entry)
            if "Timestamp" in results_df.columns:
                results_df = results_df.drop(["Timestamp","Unnamed: 0"], axis=1)
                results_df.to_csv(entry)
            total_results_df = pd.concat([total_results_df, results_df])
            dataset_name = results_df["Dataset Name"].iloc[0]
            stats_results_df = pd.DataFrame([], columns=stats_results_df_columns)
            for metric in metrics:
                metric_snap = results_df.loc[
                    (results_df["Algorithm Name"] == algorithm_tested) & (results_df["Dataset Name"] == dataset_name)][
                    metric]
                metric_other = results_df.loc[
                    (results_df["Algorithm Name"] == other_algorithm) & (results_df["Dataset Name"] == dataset_name)][
                    metric]
                try:
                    stat_less, p_less = mannwhitneyu(metric_snap, metric_other, alternative="less")
                    stat_greater, p_greater = mannwhitneyu(metric_snap, metric_other, alternative="greater")
                    stats_record_less = {"Timestamp": datetime.now(), "Metric": metric, "Dataset Name": dataset_name,
                                         "p-value": round(p_less, 3), "Less": 1}
                    stats_record_greater = {"Timestamp": datetime.now(), "Metric": metric, "Dataset Name": dataset_name,
                                            "p-value": round(p_greater, 3), "Less": 0}
                    if DEBUG_ON:
                        print('Less: Statistics=%.3f, p=%.3f' % (stat_less, p_less))
                        print('Greater: Statistics=%.3f, p=%.3f' % (stat_greater, p_greater))
                    if p_less > alpha:
                        stats_record_less.update({"Reject H0": 0})
                        if DEBUG_ON:
                            print('Less: Same distribution (fail to reject H0)')
                    else:
                        stats_record_less.update({"Reject H0": 1})
                        if DEBUG_ON:
                            print('Less: Different distribution (reject H0)')
                    if p_greater > alpha:
                        stats_record_greater.update({"Reject H0": 0})
                        if DEBUG_ON:
                            print('Greater: Same distribution (fail to reject H0)')
                    else:
                        stats_record_greater.update({"Reject H0": 1})
                        if DEBUG_ON:
                            print('Greater: Different distribution (reject H0)')
                    stats_results_df = stats_results_df.append(stats_record_less, ignore_index=True)
                    stats_results_df = stats_results_df.append(stats_record_greater, ignore_index=True)

                except Exception as err:
                    print(err)
            stats_results_df = stats_results_df.drop(["Timestamp"], axis=1)
            stats_results_df.to_csv(parent_dir_path + "/Stats Results " + dataset_name + ".csv")
            total_stats_results_df = pd.concat([total_stats_results_df, stats_results_df])

    total_stats_results_df = total_stats_results_df.drop("Timestamp", axis=1)
    total_results_df = total_results_df.drop("Timestamp", axis=1)
    total_stats_results_df.to_csv(parent_dir_path + '/Total Stats Results.csv')
    total_results_df.to_csv(parent_dir_path + '/Total Results.csv')
    total_stats_only_rejected_df = total_stats_results_df.loc[total_stats_results_df["Reject H0"] == 1]
    total_stats_only_rejected_df = total_stats_only_rejected_df.set_index(["Dataset Name", "Metric"])
    total_stats_only_rejected_df.to_csv(parent_dir_path + '/Total Rejected Stats Results.csv')
