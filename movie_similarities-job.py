#################################### Pyspark running cluster job #######################################
######################## running cluster jobs using moving rating #######################################
#This script is modified from Frank Kane course at udemy.
#The goal is to show an example of spark-submit job on a local cluster and using yarn. 
#
#AUTHORS: Frank Kane, modified by Benoit Parmentier
#DATE CREATED: unkown
#DATE MODIFIED: 08/08/2019
#Version: 1
#PROJECT: spark scaling up
#TO DO:
#
#COMMIT: clean up code for workshop
#
#################################################################################################

###### Library used in this script

import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import os 

################ NOW FUNCTIONS  ###################

##------------------
# Functions used in the script
##------------------

def loadMovieNames(in_filename):
    movieNames = {}
    with open(in_filename, encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

############################################################################
#####  Parameters and argument set up ###########

#ARGS 1
in_dir = "/home/bparmentier/Data/google_drive/Data/spark/movie_similarity_application/data/ml-100k"
#ARGS 2
out_dir = "/home/bparmentier/Data/google_drive/Data/spark/movie_similarity_application/outputs"
#ARGS3
infile_name = "u.data"
infile_name_movie_name = "u.item"

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities") #treat cores as cluster on your local machine
sc = SparkContext(conf = conf)

print("\nLoading movie names...")
nameDict = loadMovieNames(os.path.join(in_dir,infile_name_movie_name))

in_file = os.path.join(in_dir,infile_name)
data = sc.textFile(in_file) 
#type(data) #this is pyspark.rdd.RDD

# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)

# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".
if (len(sys.argv) > 1):

    scoreThreshold = 0.97
    coOccurenceThreshold = 50

    movieID=50
    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))


################################ End of script #########################################