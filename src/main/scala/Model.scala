import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

class Model {

  private val toDouble = udf[Double, String](_.toDouble)

  def train(inputPath: String, outputFolder: String) = {
    val conf = new SparkConf()
        .setMaster("local[*]")
      .setAppName("SentimentAnalysis")
    val ss = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()
    ss.sparkContext.setLogLevel("error")
    val rawDf = ss
      .read
      .option("header", true)
      .option("delimiter", "|")
      .csv(inputPath)
    val df = rawDf.withColumn("label", toDouble(rawDf("label")))
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2))
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val vectorizer = new CountVectorizer()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(30)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, vectorizer, lr))

    val model = pipeline.fit(train)
    model.write.overwrite().save(outputFolder)

    var totalCorrect = 0.0
    val result = model
      .transform(test)
      .select("prediction", "label")
      .collect()

    result.foreach{ case Row(prediction, label) => if (prediction == label) totalCorrect += 1 }
    val accuracy = totalCorrect / result.length
    println(s"Accuracy: $accuracy")

    model
  }

}

object Model extends App {
  val model = new Model().train("", "")
}
