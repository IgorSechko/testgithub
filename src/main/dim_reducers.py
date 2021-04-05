from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


NAME_CLASS = {
    "pca": PCA,
    "umap": UMAP,
    "isomap": Isomap,
    "lle": LocallyLinearEmbedding,
    "lda": LinearDiscriminantAnalysis,
    "tsne": TSNE
}


def get_reducer(name, **kwargs):
    reducer_class = NAME_CLASS[name]
    return reducer_class(**kwargs)


#
# torch.save(embeddings, "tensors/footwear_probabilities_no-scaled_umap_embedded.pt")
# embeddings = torch.load("tensors/footwear_probabilities_no-scaled_umap_embedded.pt")
