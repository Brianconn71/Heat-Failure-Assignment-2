from preprocess import returnCleanData

from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
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
    KMLabels = KMeansClustering(scaledDf, kValue)
    hierarchyLabels = hierarchialClustering(scaledDf, kValue)
    