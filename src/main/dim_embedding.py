import torch
from dim_reducers import get_reducer

if __name__ == "__main__":
    # methods = ["pca", "umap", "isomap", "lle", "lda", "tsne"]
    methods = ["isomap", "lle", "tsne"]

    net_output_files = [
        "tensors/scan_imagenet_100_footwear_output.pt",
        "tensors/scan_imagenet_200_footwear_output.pt",
        "tensors/selflabel_imagenet_100_footwear_output.pt",
        "tensors/selflabel_imagenet_200_footwear_output.pt"
    ]

    for file in net_output_files:
        print("input:", file)
        _, features = torch.load(file)

        for method in methods:
            print("method:", method)
            try:
                reducer = get_reducer(method, n_components=2)
                embeddings = reducer.fit_transform(X=features)

                save_name = f"{file[:-3]}_{method}_embedded.pt"
                torch.save(embeddings, save_name)
            except:
                print("an exception occurred")
