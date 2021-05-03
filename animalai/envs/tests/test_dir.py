from cognitive.evaluate import load_dir_dataset


def test_split():
    datasets = load_dir_dataset()
    for name, ds in datasets.items():
        train, val = ds.split(0.1)
        return
