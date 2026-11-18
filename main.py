from preprocess import returnCleanData

from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import pandas as pd

# setting display settings to ensure columns are not cut off when dispalyed
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

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

def KMeansClustering(scaledDf, kValue):
    # Creating the kmeans model with k input from the above findBestKValue function
    # using same random_state and n_init as findBestKValue for consistency of results
    kMeansModel = KMeans(n_clusters=kValue, random_state=42, n_init=25)

    # fit_predict methos is used to run the kmeans algorithm and then returns a cluster label for each patient in the study
    # fit_predict finds the cluster centres and uses this to predict a label for each patient in our dataset.
    # the reurn is of type array
    KMLabels = kMeansModel.fit_predict(scaledDf)

    # using print statements to print to the terminal How many patients are in ach cluster
    # using value_counts pandas method to count how many patients have each label
    print("\n--------------K-Means Cluster Size-----------------\n")
    print(pd.Series(KMLabels).value_counts().sort_index())

    # using print statements to print to the terminal the scaled cluster centres and showing which feature separates the clusters
    # cluster centres has the cordinates of each cluster centre point returned as an array
    # continuous columns get scaled and they are displayed as standard deviations from the mean whereas
    # binary columns are proportional and not scaled example being anemia which is 0.41 which means
    # 41% of cluster 0 has anemia
    print("\n--------------K-Means Cluster Centers Scaled-----------------\n")
    centreValues = pd.DataFrame(kMeansModel.cluster_centers_, columns=scaledDf.columns)
    print(centreValues.round(2))

    # we then need to return the kmeans labels for evaluation later
    return KMLabels

def hierarchialClustering(scaledDf, kValue):
    # hierarchial is agglomerative which means bottoms up and in this case each patient will start as his own cluster
    # the two closest clusters are then merged until eventualy kValue clusters are all that are left.
    # linkage is used to merge the pair that increases the within-cluster variance the least of alland is similar to what k-means minimizes , which means that 
    # the two methods can be fairly compared. If the linkage is set as ward then only euclidean and l2 are accepted.
    # unlike kmeans hierarchial clustering does not take random_state argument so hierarchial always returns the same results
    hierarchyModel = AgglomerativeClustering(n_clusters=kValue, linkage="ward")

    # fit_predict method is used to run the hierarchial algorithm and then returns a cluster label for each patient in the study
    hierarchyLabels = hierarchyModel.fit_predict(scaledDf)

    print("\n--------------Hierarchial Cluster Size-----------------\n")
    print(pd.Series(hierarchyLabels).value_counts().sort_index())

    # we need a way to visualize this and we start by getting the full merge history using the same mehod ("ward")
    # as what was used on the model
    merge = linkage(scaledDf, method="ward")

    cuttingHeight = (merge[-kValue,2] + merge[-(kValue - 1), 2]) / 2

    # we need the figure to be drawn before plotting the dendrogram
    plt.figure(figsize=(12, 6))

    # drawing and intitaiting the dendrogram method 
    # we are using p=50 and truncate_mode=lastp which means to show the last 50 merges done on the dataframe
    dendrogram(merge, truncate_mode="lastp", p=50, color_threshold=cuttingHeight)

    
    plt.title("Hierarchial Clustering Dendrogram using Ward Linkage")
    plt.xlabel("Patients in the study (cluster size in brackets)")
    plt.ylabel("Merge distance")
    
    plt.axhline(y=cuttingHeight, color="red", linestyle="--", label=f"cut at k={kValue}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figurePath / "dendrogram.png", dpi=250)
    plt.show()

    return hierarchyLabels

def plotClusters(scaledDf, KMLabels, hierarchyLabels, kValue):
    pca = PCA(n_components=2)
    pcaFitted = pca.fit_transform(scaledDf)

    variance = pca.explained_variance_ratio_
    print(f"\nPCA: PC1 {variance[0]*100:.2f}%, PC2: {variance[1]*100:.2f}%, Total: {variance.sum()*100:.2f}%")

    components = pd.DataFrame(pca.components_, columns=scaledDf.columns, index=["PC1", "PC2"])
    print(components.T.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)

    for axis, labels, title in zip(axes, [KMLabels, hierarchyLabels], ["K-Means Algorithm", "Hierarchial Algorithm (using ward linkage)"]):
        scatter = axis.scatter(pcaFitted[:, 0], pcaFitted[:, 1], c=labels, cmap="coolwarm", s=20, alpha=0.8, edgecolor="white", linewidth=0.5)
        axis.set_title(f"{title} (k={kValue})")
        axis.set_xlabel(f"PC1 {variance[0]*100:.2f}% of variance")
        axis.set_ylabel(f"PC2 {variance[1]*100:.2f}% of variance")
        axis.legend(*scatter.legend_elements(), title="Cluster")

    plt.tight_layout()
    plt.savefig(figurePath / "clusters.png", dpi=250)
    plt.show()

def calculateTheWcss(scaledDf, labels):
    # k-means has WCSS via inertia_ mehod but hierarchial does not with agglomerative clustering
    # wcss needs to be caalculated manually for this and used on both of the algorithms for fairness
    # WCSS is the total squared distance each patient in the study is to the centrre of their cluster.

    total = 0

    # sort and ensure only unique values in the labels list
    for cluster in sorted(set(labels)):
        # labels equaling cluster gives a True or False value for each patient and keeps only these patients in thaat cluster.
        points = scaledDf[labels == cluster]
        # centre of the cluster is the mean value for patients
        centre = points.mean()
        # get the squared distance of every patient to the centre, first sum is for each column and second is for total of all columns together.
        total += ((points - centre) ** 2).sum().sum()

    # return the Wcss
    return total


def evaluateClusters(scaledDf, KMLabels, hierarchyLabels):
    results = []

    for name, label in [("K-Means", KMLabels), ("Hierarchial (by ward linkage)", hierarchyLabels)]:
        results.append({
            "Algorithm": name,
            "Silhouette": silhouette_score(scaledDf, label),
            "Davies-Bouldin": davies_bouldin_score(scaledDf, label),
            "WCSS": calculateTheWcss(scaledDf, label)
        })

    resultsDf = pd.DataFrame(results).set_index("Algorithm")

    print("\n--------------Cluster Evaluation-----------------\n")
    print(resultsDf.round(3))
    
    return resultsDf

def compareClusters(KMLabels, hierarchyLabels):
    # Adj rand index is used to compare every pair of patients/participants and check if both algorithms put them together or pull apart
    # 1 = both algorithms give the same cluster, 0 = chance but no better than that
    # it ignores label numbers when comparing so if cluster 0 in one is 1 in the other
    adjRand = adjusted_rand_score(KMLabels, hierarchyLabels)

    # crosstab - shows k-means clusters as rows and hierarchy clusters as columns
    # diagonal is where both algorithms agree and off diagonal is where they disagree
    # margins = True is used to add an All row and column that has totals
    crossTab = pd.crosstab(KMLabels, hierarchyLabels, rownames=["K-Means"], colnames=["Hierarchial"], margins=True)
    print(crossTab)

    # inAgreement is the percent of patients put in the same cluster by both algorithms
    # cluster numbers can be random so we need to check swapped labels but keep higher value.
    # this will only work while k is 2
    sameLabel = (KMLabels == hierarchyLabels).mean()
    inAgreement = max(sameLabel, 1 - sameLabel)


    print("\n--------------Comparing K-Means with Hierarchial-----------------\n")
    print(f"Adjusted Rand Index: {adjRand:.3f}")
    print(f"Patients in same cluster in both algorithms {inAgreement*100:.2f} %")
    print(crossTab)

    return adjRand, crossTab

def profileClusters(unscaledDf, KMLabels, hierarchyLabels):
    binaryCols = [col for col in unscaledDf.columns if unscaledDf[col].nunique() == 2]

    profiles = {}

    for name, label in [("K-Means", KMLabels), ("Hierarchy", hierarchyLabels)]:
        unScale = unscaledDf.copy()
        unScale["Cluster"] = label

        profile = unScale.groupby("Cluster").mean().T

        profile["All Patients"] = unscaledDf.mean()

        profile.loc[binaryCols] = profile.loc[binaryCols] * 100

        sizes = unScale["Cluster"].value_counts().sort_index()
        profile.loc["Patients (n)"] = list(sizes) + [len(unScale)]

        print(f"\n--------------{name} Cluster Profiles -----------------\n")
        print(profile.round(2))

        profiles[name] = profile
    return profiles


if __name__ == "__main__":
    # call the return clean data function within preprocess each time this file is run so we know the data is accurate
    unscaledDf, scaledDf = returnCleanData()
    kValue = findBestKValue(scaledDf)
    KMLabels = KMeansClustering(scaledDf, kValue)
    hierarchyLabels = hierarchialClustering(scaledDf, kValue)
    plotClusters(scaledDf, KMLabels, hierarchyLabels, kValue)
    results = evaluateClusters(scaledDf, KMLabels, hierarchyLabels)
    adjRand, crossTab = compareClusters(KMLabels, hierarchyLabels)
    profiles = profileClusters(unscaledDf, KMLabels, hierarchyLabels)
    