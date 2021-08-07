import pandas as pd

def load_data(name, dataset_dict):
    path = dataset_dict["path"]
    target = dataset_dict["target"]
    df = pd.read_csv(path)
    if name == "Analcatdata Boxing":
        df[target] = df[target].map({"Holyfield": 1, "Lewis": 0})
    elif name in ["Bodyfat", "Cloud", "Chatfield", "Disclosure", "Diggle", "Kidney", "Visualizing Livestock", "Veteran",
                  "Socmob", "Schlvote", "PM10", "Plasma Retinol", "Meta", "NO2"]:
        df[target] = df[target] = df[target].map({"P": 1, "N": 0})
        if name == "Kidney":
            df = df.drop("patient", axis=1)
        elif name == "Meta":
            df = df.drop(["cancor2", "fract2"], axis=1)
            df = df.dropna()
    elif name == "Diabetes":
        df[target] = df[target] = df[target].map({"tested_positive": 1, "tested_negative": 0})
    X, y = df.drop(target, axis=1), df[target]
    return X, y, df
