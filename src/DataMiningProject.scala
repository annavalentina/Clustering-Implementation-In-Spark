/* DataMiningProject.scala */
import scala.io.Source
import java.io._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.util.control._
import org.apache.log4j.Logger
import org.apache.log4j.Level

object DataMiningProject{
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    // Create spark configuration
    val sparkConf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("DataMining")
    val sc =  new SparkContext(sparkConf)



    //Get file directory
    println("Give input file: ")
     val filename = scala.io.StdIn.readLine()
    


    /*Get the lines from the file and filter out those than don't contain "," and those that start or end with ",".This will
    get rid of the points that don't have both values.
    Then split each line in half.
     After that an Array of two empty lists is created and then filled with the points so that in the end we will have a list for all the x and one for   all the y of the points.*/
    val pair: (List[Double], List[Double]) = Source.fromFile(filename).getLines().filter(line => line.contains(",")).filter(line => !(line.startsWith(","))).filter(line => !(line.endsWith(","))).map(line => line.split(",").map(_.toDouble)).filter(_.nonEmpty).foldLeft((List[Double](), List[Double]()))((acc, cur) => (cur(0) :: acc._1, cur(1) :: acc._2))


    /*The two lists of the array are splitted.
    The lists were in reverse order than it was in the file*/
    val leftList: List[Double] = pair._1.reverse
    val rightList: List[Double] = pair._2.reverse

    //Find min-max of each list and normalizes them using the MinMax normalization
    var min = leftList.reduceLeft(_ min _)
    var max = leftList.reduceLeft(_ max _)
    val newLeft = leftList.map(line => (1 - 0) / (max - min) * (line - min) + 0)

    min = rightList.reduceLeft(_ min _)
    max = rightList.reduceLeft(_ max _)
    val newRight = rightList.map(line => (1 - 0) / (max - min) * (line - min) + 0)

    //The two lists are merged and parallelized into one RDD
    val parsedData = sc.parallelize(newLeft.zip(newRight)).map(_.toString.replaceAll("[()]", "")).map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
     
    println("Application started...")

  /*KMeans starts*/
    var numClusters = 1
    //Variable to hold current number of clusters
    val numIterations = 40
    // Internal iterations
    val numExternalIterations = 5
    //External iterations
    var maxChange = 0.0
    var change = 0.0
    var lastSSE = 0.0
    //SSE of previous numClusters
    var bestSSE = 0.0
    //Minimum SSE of numClusters
    var bestK = 1
    //Ideal number of clusters
    val threshold = 10.0
    val loop = new Breaks
    var done = false
    var WSSSE = 0.0

    //For numClusters=1
    var clusters = KMeans.train(parsedData, numClusters, numIterations)
    bestSSE = clusters.computeCost(parsedData) //Find SSE

    //Execute KMeans 5 times and finds the minimum SSE. This is done to prevent errors due to random selection of the initial centers
    for (i <- 2 to numExternalIterations) {
      clusters = KMeans.train(parsedData, numClusters, numIterations)
      WSSSE = clusters.computeCost(parsedData)
      //Keep the smaller SSE
      if (bestSSE > WSSSE)
        bestSSE = WSSSE
    }

    loop.breakable {
      while (!done) {
        if (numClusters != 1) {
          change = ((lastSSE - bestSSE) * 100) / bestSSE //Find percentage of change between SSE of current numClusters and numClusters-1
          //Keep the maximum change
          if (change > maxChange) {
            bestK = numClusters //This number of clusters is the best so far
            maxChange = change
          }
          //If change is smaller than the threshold the loop breaks
          if (change < threshold) {
            loop.break
          }
        }

        numClusters += 1
        lastSSE = bestSSE
        //Execute KMeans 5 times and finds the minimum SSE.
        for (i <- 1 to numExternalIterations) {
          clusters = KMeans.train(parsedData, numClusters, numIterations)
          WSSSE = clusters.computeCost(parsedData)
          if (i == 1)
            bestSSE = WSSSE
          else if (bestSSE > WSSSE)
            bestSSE = WSSSE

        }

        //If SSE is 0.0 then the clusters are as many as the points. That isn't optimal so the loop stops
        if (bestSSE == 0.0) {
          done = true
        }

      }
    }

    //Execute the KMeans for the best number of clusters
    for (i <- 1 to numExternalIterations) {
      clusters = KMeans.train(parsedData, bestK, numIterations)
      WSSSE = clusters.computeCost(parsedData)
      if (i == 1)
        bestSSE = WSSSE
      else if (bestSSE > WSSSE)
        bestSSE = WSSSE

    }


    val clustersWithIds = sc.parallelize(clusters.clusterCenters.zipWithIndex).map(x => (x._2, x._1)) //Give Id to cluster centers
    //Print results
    println("----------KMeans Results----------")
    println("Number of clusters: " + bestK)
    for (i <- clustersWithIds)
      println("Cluster: " + i._1 + " Center: " + i._2)


    /*Silhouette Coefficient starts*/

    val vectorsAndClusterIdx = parsedData.map { point =>
      val prediction = clusters.predict(point)
      (point, prediction)
    }
    val clustersWithIndex = sc.parallelize(clusters.clusterCenters.zipWithIndex)

    //Get the cartesian product of points X clusters(centroid)
    val cartesianPointsCentres = vectorsAndClusterIdx.cartesian(clustersWithIndex)

    //For every pair (point, centroid) calculate the euclidean distance
    val distRDD = cartesianPointsCentres.map(x => (x._1, x._2, Math.sqrt(Math.pow(x._1._1(0) - x._2._1(0), 2) + Math.pow(x._1._1(1) - x._2._1(1), 2)))) //Returns (point,centroid,distance)


    //To calculate A, first we filter those pairs that the point and centroid belong to the same cluster
    val Ai = distRDD.filter(x => (x._1._2 == x._2._2)).map(x => (x._1, x._3)) //Returns (point,distance) where distance is Ai

    //To calculate B, first we filter those pairs that the point and centroid do NOT belong to the same cluster
    val Bi = distRDD.filter(x => (x._1._2 != x._2._2)).map(x => (x._1, x._3)).reduceByKey((a, b) => (if (a < b) a else b)) //Returns (point,distance) where distance is Bi

    val joinedAB = Ai.join(Bi) //Return (point,ai,bi)

    //Now we can calculate the Si for every point and  map all the Si's by the cluster they belong to and reduce to get the sum of the Si's
    val readyRDD = joinedAB.map(x => (x._1._2, (x._2._2 - x._2._1) / Math.max(x._2._1, x._2._2))) //Returns (fromCluster,Si)
      .reduceByKey(_ + _) //Returns (fromCluster,SiSum)

    //Map-Reduce to calculate th number of points that belong to each cluster
    val counterRDD = joinedAB.map(x => (x._1._2, 1)).reduceByKey(_ + _) //Returns (fromCluster,counter)

    val bothSiCounter = readyRDD.join(counterRDD) //Returns (cluster,SiSum,counter)

    //Calculate average Si for each cluster
    val clusterSiRDD = bothSiCounter.map(x => (x._1, x._2._1 / x._2._2)) //Returns (cluster,SiAVG)

    //Print
    println("--------Silhouette Results--------")
    for (cluster <- clusterSiRDD)
      println("Silhouette  coefficient of cluster: " + cluster._1 + " is: " + cluster._2)


    /*Outliers searching starts*/
    /*To find the anomalies we need the Ai RDD from the silhouette coefficient since we only need the distance from every point to the centroid they belong to
    and the counterRDD where we have already calculated the number of points every cluster has*/

    val dists = Ai.map(x => (x._1._2, x._2)) //Returns (cluster, distance)

    //Calculate the mean distance for every cluster
    val mean = dists.reduceByKey(_ + _) //Get the sum of all the distances on the cluster
      .join(counterRDD).map(x => (x._1, x._2._1 / x._2._2)) //Calcuate the mean for every cluster, returns (cluster, mean distance)

    //Calcuate the standard deviation for every cluster
    val sd = mean.join(dists).map(x => (x._1, Math.pow(x._2._2 - x._2._1, 2))).reduceByKey(_ + _) //Get the sum of every (distance-mean)^2
      //Calculate the standard deviation and in case there is only one point in a cluster set the number of points to 1
      .join(counterRDD).map(x => (x._1, Math.sqrt(x._2._1 / (if ((x._2._2 - 1) > 0.0) (x._2._2 - 1) else 1.0)))) //returns (cluster,sd)

    //In order to detect the anomalies we will look for points that their distance does not belong on [mean-3sd, mean+3sd]
    val anomalies = Ai.map(x => (x._1._2, (x._1._1, x._2))).join(sd).join(mean) //Returns (cluster,(((point,distance),mean), sd))
      .filter(x => ((x._2._1._1._2 - x._2._2) > 3 * x._2._1._2) || ((x._2._1._1._2 - x._2._2) < (-3) * x._2._1._2)) //Get only those points that do not belong on the given range
      .map(x => (x._1, x._2._1._1._1))

    //Print
    println("-------------Outliers-------------")
    for (point <- anomalies)
      println("Point: " + point._2 + " from cluster: " + point._1)

    sc.stop()
  }

}
