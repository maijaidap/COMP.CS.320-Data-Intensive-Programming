package assignment22

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{DoubleType, LongType, Metadata, StringType, StructField, StructType}


class Assignment {

  val spark: SparkSession = SparkSession.builder()
    .appName("Assignment")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  // Schema for tasks 1 and 4
  // Defining schemas improve performance especially with large files.
  val schemaAB = StructType(Array(
    StructField("a", DoubleType, true),
    StructField("b", DoubleType, true),
    StructField("LABEL", StringType, true))
  )

  // Schema for task 2
  val schemaABC = StructType(Array(
    StructField("a", DoubleType, true),
    StructField("b", DoubleType, true),
    StructField("c", DoubleType, true),
    StructField("LABEL", StringType, true))
  )

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read
    .option("sep", ",")
    .option("header", "true")
    .schema(schemaAB)
    .csv("data/dataD2.csv")

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read
    .option("sep", ",")
    .option("header", "true")
    .schema(schemaABC)
    .csv("data/dataD3.csv")

  // Transforms the LABEL column data into numeric values
  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = dataD2.withColumn("numeric_label", when(col("LABEL")
    .contains("Ok"), 0).otherwise(1))

  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    // Transforms the input data into a vector form
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")

    // Apply min max scaling
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")

    // Create a pipeline from vector assembler and scaler, then transform a new dataframe
    val transformationPipeline = new Pipeline().setStages( Array(vectorAssembler, scaler) )
    val transformedData = transformationPipeline.fit(df).transform(df)

    // Get KMeans and fit data
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val kmeansModel = kmeans.fit(transformedData)

    // Get cluster centers
    val clusterCenters = kmeansModel.clusterCenters.map(x => (x(0), x(1)))

    clusterCenters
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    // Transforms the input data into a vector form
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")

    // Apply min max scaling
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")

    // Create a pipeline from vector assembler and scaler, then transform a new dataframe
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler, scaler))
    val transformedData = transformationPipeline.fit(df).transform(df)

    // Get KMeans and fit data
    val kmeans = new KMeans().setK(k).setFeaturesCol("scaled_features")
    val kmeansModel = kmeans.fit(transformedData)

    // Get cluster centers
    val clusterCenters = kmeansModel.clusterCenters.map(x => (x(0), x(1), x(2)))

    clusterCenters
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    // Transforms the input data into a vector form
    val vectorAssembler = new VectorAssembler()
      .setInputCols( Array("a","b","numeric_label") )
      .setOutputCol("features")

    // Apply min max scaling
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")

    // Create a pipeline from vector assembler and scaler, then transform a new dataframe
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler, scaler))
    val transformedData = transformationPipeline.fit(df).transform(df)

    // Get KMeans and fit data
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val kmModel = kmeans.fit(transformedData)

    // Get cluster centers and return cluster centers for fatal values.
    val clusterCenters = kmModel.clusterCenters
    clusterCenters.foreach(x => print(x))
    val filterValues = clusterCenters.filter(x => x(2) >= 0.2)

    filterValues.map(x => (x(0), x(1)))
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    // Transforms the input data into a vector form
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features_unscaled")

    // Apply min max scaling for features column
    val scaler = new MinMaxScaler()
      .setInputCol("features_unscaled")
      .setOutputCol("features")

    // Create a pipeline from vector assembler and scaler, then transform a new dataframe
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler, scaler))
    val transformedData = transformationPipeline.fit(df).transform(df)

    // Calculating total number of clusters and silhouette scores
    val scores = countClustersAndSilhouettes(transformedData, low, high)

    scores
  }

  // Recursive function for counting the amount of clusters and silhouette score for clustering
  // Used for task 4.
  def countClustersAndSilhouettes(df: DataFrame, low: Int, high: Int): Array[(Int, Double)] = {

    // Get KMeans and fit data
    val kmeans = new KMeans().setK(low).setSeed(1L)
    val kmModel = kmeans.fit(df)

    // Make predictions
    val predictions = kmModel.transform(df)

    // Compute silhouette score
    val silhouetteScore = new ClusteringEvaluator().evaluate(predictions)

    // Recursively counting amount of clusters and silhouette score for clustering
    if (low == high) {
      return Array((low, silhouetteScore))
    } else {
      return Array((low, silhouetteScore)) ++: countClustersAndSilhouettes(df, low + 1, high)
    }

  }
}
