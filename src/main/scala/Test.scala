import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import util.Stopwatch



object Test extends App with Stopwatch {
  private val savedModelFolder = ???
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("SentimentAnalysisTest")
  val ss = SparkSession
    .builder()
    .config(conf)
    .getOrCreate()
  ss.sparkContext.setLogLevel("error")

  val df = ss.read
    .option("header", true)
    .csv("./test.csv")
  df.show()
  val pipelineModel = PipelineModel.read.load("./twitter_sentiment_model")
  time {
    pipelineModel.transform(df).foreach(r => println(s"Text: ${r(1)}, Score: ${r(r.length - 1)}"))
  }
}
