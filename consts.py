parent_dir_path = "Results"

other_algorithm = "RandomForest"
algorithm_tested = "Snapshot Ensemble"
HYPER_TUNING = True
CV_OFF = False
DEBUG_ON = False
alpha = 0.05
num_of_iter = 50

param_grid_snap = {
    'batch_size': [32],
    'lr': [.01],
    'ephocs_cycle': [20, 30, 50, 70, 100],
    'M_models': [15, 20],
    'N_top_Models': [3, 5, 7, 10, 12]
}
param_grid_other = {
    'n_estimators': [2, 5, 10, 50, 100],
    'max_depth': [None, 10, 20, 50, 70],
    'max_features': ["sqrt", "log2"],
}

datasets_dicts = {
    "Analcatdata Boxing": {"path": "classification-datasets/analcatdata_boxing1.csv",
                           "target": "Winner",
                           "categorical_integers": ["Round"],
                           "categorical_strings": ["Judge", "Official"],
                           },  # hard
    "Blood": {"path": "classification_datasets/blood.csv",
              "target": "clase",
              "categorical_integers": [],
              "categorical_strings": [],
              },
    "Bodyfat": {"path": "classification_datasets/bodyfat.csv",
                "target": "binaryClass",
                "categorical_integers": ["Age"],
                "categorical_strings": [],
                },
    "Breast Cancer": {"path": "classification_datasets/breast-cancer.csv",
                      "target": "clase",
                      "categorical_integers": [],
                      "categorical_strings": [],
                      },
    "Cloud": {"path": "classification_datasets/cloud.csv",
              "target": "binaryClass",
              "categorical_integers": [],
              "categorical_strings": ["SEEDED"],
              },
    "Chatfield": {"path": "classification_datasets/chatfield_4.csv",
                  "target": "binaryClass",
                  "categorical_integers": ["col_1"],
                  "categorical_strings": [],
                  },
    "Diabetes": {"path": "classification_datasets/diabetes.csv",
                 "target": "class",
                 "categorical_integers": ["preg", "plas", "pres", "skin", "insu", "age"],
                 "categorical_strings": [],
                 },  # hard
    "Diggle": {"path": "classification_datasets/diggle_table_a2.csv",
               "target": "binaryClass",
               "categorical_integers": ["col_1", "col_2", "col_3", "col_4", "col_5"],
               "categorical_strings": [],
               },
    "Kidney": {"path": "classification_datasets/kidney.csv",
               "target": "binaryClass",
               "categorical_integers": ["age", "status", "time"],
               "categorical_strings": ["sex", "disease_type"],
               },  # hard

    "Visualizing Livestock": {
        "path": "classification_datasets/visualizing_livestock.csv",
        "target": "binaryClass",
        "categorical_integers": [],
        "categorical_strings": ["livestocktype", "country"],
    },
    "Veteran": {"path": "classification_datasets/veteran.csv",
                "target": "binaryClass",
                "categorical_integers": ["therapy", "age", "months", "karnofsky", "status", "celltype", "treatment"],
                "categorical_strings": [],
                },
    "Statlog Heart": {"path": "classification_datasets/statlog-heart_.csv",
                      "target": "clase",
                      "categorical_integers": [],
                      "categorical_strings": [],
                      },
    "Statlog Australian Credit": {
        "path": "classification_datasets/statlog-australian-credit.csv",
        "target": "clase",
        "categorical_integers": [],
        "categorical_strings": [],
    },
    "Socmob": {"path": "classification_datasets/socmob.csv",
               "target": "binaryClass",
               "categorical_integers": [],
               "categorical_strings": ["race", "family_structure", "sons_occupation", "fathers_occupation"],
               },
    "Prnn Synth": {"path": "classification_datasets/prnn_synth.csv",
                   "target": "yc",
                   "categorical_integers": [],
                   "categorical_strings": [],
                   },
    "PM10": {"path": "classification_datasets/pm10.csv",
             "target": "binaryClass",
             "categorical_integers": ["hour_of_day"],
             "categorical_strings": [],
             },  # TO KEEP
    "Plasma Retinol": {"path": "classification_datasets/plasma_retinol.csv",
                       "target": "binaryClass",
                       "categorical_integers": ["AGE", "BETADIET", "RETDIET", "BETAPLASMA"],
                       "categorical_strings": ["VITUSE", "SMOKSTAT", "SEX"],
                       },
    "Meta": {"path": "classification_datasets/meta.csv",
             "target": "binaryClass",
             "categorical_integers": ["T", "N", "p", "k", "Bin", "Cost"],
             "categorical_strings": ["Alg_Name", "DS_Name"],
             },
    "NO2": {"path": "classification_datasets/no2.csv",
            "target": "binaryClass",
            "categorical_integers": ["hour_of_day"],
            "categorical_strings": [],
            },  # TO KEEP
    "Pima": {"path": "classification_datasets/pima.csv",
             "target": "clase",
             "categorical_integers": [],
             "categorical_strings": []
             },
}
metrics = ["Accuracy", "TPR", "FPR", "Precision", "AUC", "AUPRC", "Training Time", "Inference Time"]
stats_results_df_columns = ["Timestamp", "Dataset Name", "Metric", "p-value", "Reject H0", "Less"]
results_df_columns = ["Timestamp", "Dataset Name", "Cross Validation", "Algorithm Name"] + metrics
