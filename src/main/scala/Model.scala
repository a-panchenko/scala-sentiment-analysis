import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import util.Stopwatch


class Model extends Stopwatch {
  def train(inputPath: String, outputFolder: String): PipelineModel = {
    // Setup SparkSession
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("SentimentAnalysis")
    val ss = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()
    ss.sparkContext.setLogLevel("error")
    time {
        // Read raw dataset
        val rawDf = ss
          .read
          .option("header", true)
          .option("delimiter", ",")
          .csv(inputPath)
        rawDf.persist()
        rawDf.show(10)
        println(s"Number of rows: ${rawDf.count()}")

        // Basic data cleaning
        val df = rawDf.withColumn("label", toDouble(rawDf("Sentiment")))
          .withColumn("text", rawDf("SentimentText"))
          .drop("SentimentSource", "\uFEFFItemID", "SentimentText", "Sentiment")
        df.show(10)

        df.withColumn("text_length", length(df("text")))
          .groupBy("label")
          .avg("text_length")
          .toDF("label", "avg_text_length")
          .show()
      // Split data on the train and test datasets
      val Array(train, test) = df.randomSplit(Array(0.8, 0.2))

      // Use tokenizer for splitting text on the words
      val tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words")

      // Use CountVectorizer for converting words into vectors of integers
      val vectorizer = new CountVectorizer() //
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("features")

      // Logistic regression is one of the most simple algorithms for classification problems
      val lr = new LogisticRegression()
        .setRegParam(0.001)

      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, vectorizer, lr))

      // Train the model
      val model = pipeline.fit(train)
      model.write.overwrite().save(outputFolder)

      // Evaluate the model
      var totalCorrect = 0.0
      val result = model
        .transform(test)
        .select("prediction", "label")
        .collect()

      result.foreach { case Row(prediction, label) => if (prediction == label) totalCorrect += 1 }
      val accuracy = totalCorrect / result.length
      println(s"Accuracy: $accuracy")

      model
    }
  }

  private val toDouble = udf[Double, String](_.toDouble)
  private val length = udf[Long, String](_.length)

}

object Model extends App {
  val model = new Model().train("", "")
}
