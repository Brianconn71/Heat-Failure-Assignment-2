from preprocess import returnCleanData

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

figurePath = Path(__file__).parent / "figures"

def findBestKValue(scaledDf):
    # Need to find best k to use on the clustering algorithms
    # I am going to compare the WCSS method and the silhouette method here
    # references added to report

    # silhouette requires a minimum of 2 clusters so range starts at 2
    # keep range rather low as it needs to be clinically feasible and anthing over 15 is too high
    kValues = range(2, 16)
    wcss = []
    silhouettes = []

    for k in kValues:

        # random_state fixes the random starting centres so results are coonsistent
        # n_init=25 means k-means runs 25 times with different starts but keeps best results
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
        kmeans.fit(scaledDf)

        # scikit learn uses inertia as a means of wcss
        wcss.append(kmeans.inertia_)

        # labels is the cluster number assigned to the participant
        silhouettes.append(silhouette_score(scaledDf, kmeans.labels_))

        print(f"k={k}: WCSS={kmeans.inertia_:.2f}, silhouette={silhouettes[-1]:.3f}")

    # printing the best values for k in both wcss and silhouttes
    bestSilhouettesIdx = silhouettes.index(max(silhouettes))
    bestWcssIdx = wcss.index(max(wcss))

    bestSilhouetteK = kValues[bestSilhouettesIdx]
    bestWcssK = kValues[bestWcssIdx]

    print(f"Best K value for Silhouettes: {bestSilhouetteK}")
    print(f"Best K value for WCSS: {bestWcssK}")

    # Plotting the scores to a side by side graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(kValues, wcss, marker="x")
    ax1.set_xlabel("Number of k clusters")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method")
    ax1.axvline(bestWcssK, color="red", linestyle="--")

    ax2.plot(kValues, silhouettes, marker="x")
    ax2.set_xlabel("Number of k clusters")
    ax2.set_ylabel("Silhouette")
    ax2.set_title("Silhouette Score")
    ax2.axvline(bestSilhouetteK, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(figurePath / "K_Values.png", dpi=250)
    plt.show()

    # as we have shown above the best k value from both means of finding the best k is 2 so I will return bestSilhouetteK but it does not matter as both values are samne
    return bestSilhouetteK

def KMeansClustering():
    return

def hierarchialClustering():
    return

def plotClusters():
    return

def evaluateClusters():
    return

def compareClusters():
    return

def profileClusters():
    return


if __name__ == "__main__":
    # call the return clean data function within preprocess each time this file is run so we know the data is accurate
    unscaledDf, scaledDf = returnCleanData()
    kValue = findBestKValue(scaledDf)
    