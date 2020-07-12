package com.isuru.analyzer;

import com.isuru.bean.Comment;
import com.isuru.bean.NewsArticle;
import com.isuru.bean.Sentiment;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Extract comments from NewsArticle.
 */
public class KappaCalculator {
    private static final Logger logger = Logger.getLogger("Aggregator");

    private static final String SEPARATOR = ";";
    private static final String NEW_LINE = "\n";
    private StringBuilder aggregateComments = new StringBuilder().append("docid;comment;label\n");

    private static List<String> fileNames = Arrays.asList(
            "44969.txt", "44974.txt", "44992.txt", "45272.txt", "45483.txt", "45871.txt",
            "100204.txt", "100475.txt", "101559.txt", "101944.txt", "102756.txt", "103990.txt",
            "105789.txt", "105812.txt", "107019.txt", "107342.txt", "107507.txt", "108192.txt",
            "108684.txt", "113922.txt", "114780.txt", "115652.txt", "118450.txt", "121120.txt",
            "122150.txt", "122704.txt", "123076.txt", "124564.txt", "124680.txt", "125388.txt"
    );

    List<String> commentEntryList = new ArrayList<>();

    private static int positivePositive = 0;
    private static int positiveNegative = 0;
    private static int positiveNeutral = 0;
    private static int negativePositive = 0;
    private static int negativeNegative = 0;
    private static int negativeNeutral = 0;
    private static int neutralPositive = 0;
    private static int neutralNegative = 0;
    private static int neutralNeutral = 0;
    private static int other = 0;

    public static void main(String[] args) {
        String folder1String = "./corpus/kappa/1/";
        String folder2String = "./corpus/kappa/2/";


        KappaCalculator kappaCalculator = new KappaCalculator();

        for (String s : fileNames) {
            File file1 = new File(folder1String + s);
            File file2 = new File(folder2String + s);

            kappaCalculator.processFiles(file1, file2);

        }

        System.out.println("positivePositive = " + positivePositive);
        System.out.println("positiveNegative = " + positiveNegative);
        System.out.println("positiveNeutral = " + positiveNeutral);
        System.out.println("negativePositive = " + negativePositive);
        System.out.println("negativeNegative = " + negativeNegative);
        System.out.println("negativeNeutral = " + negativeNeutral);
        System.out.println("neutralPositive = " + neutralPositive);
        System.out.println("neutralNegative = " + neutralNegative);
        System.out.println("neutralNeutral = " + neutralNeutral);
        System.out.println("other = " + other);

        int agree = positivePositive + negativeNegative + neutralNeutral;
        int all = positivePositive + positiveNegative + positiveNeutral +
                negativePositive + negativeNegative + negativeNeutral +
                neutralPositive + neutralNegative + neutralNeutral +
                other;

        int rater1Positive = positivePositive + positiveNegative + positiveNeutral;
        int rater1Negative = negativePositive + negativeNegative + negativeNeutral;
        int rater1Neutral = neutralPositive + neutralNegative + neutralNeutral;
        int rater2Positive = positivePositive + negativePositive + neutralPositive;
        int rater2Negative = positiveNegative + negativeNegative + neutralNegative;
        int rater2Neutral = positiveNeutral + negativeNeutral + neutralNeutral;


        double po = agree * 1.0 / all;
//        double pe = 1.0 / all /all * (rater1Positive * rater2Positive + rater1Negative * rater2Negative + rater1Other * rater2Other);
        double pe = 1.0 / all /all * (rater1Positive * rater2Positive + rater1Negative * rater2Negative + rater1Neutral * rater2Neutral);
        double kappa = (po - pe) * 1.0 / (1 - pe);

        System.out.println("Total comments =  " + all);
        System.out.println("Po = " + po + ", " + "Pe = " + pe);
        System.out.println("Agreement percentage = " + agree * 1.0 / all);
        System.out.println("kappa = " + kappa);

        kappaCalculator.writeToFiles();
    }

    private List<String> processFiles(File file1, File file2) {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
            NewsArticle newsArticle1 = (NewsArticle) jaxbUnmarshaller.unmarshal(file1);
            NewsArticle newsArticle2 = (NewsArticle) jaxbUnmarshaller.unmarshal(file2);

            for (int i = 0; i < newsArticle1.getComments().size(); i++) {
                Comment comment1 = newsArticle1.getComments().get(i);
                Comment comment2 = newsArticle2.getComments().get(i);

                if (comment1.getSentiment().equals(Sentiment.POSITIVE)) {
                    if (comment2.getSentiment().equals(Sentiment.POSITIVE)) {
                        positivePositive++;
                    } else if (comment2.getSentiment().equals(Sentiment.NEGATIVE)) {
                        positiveNegative++;
                    }  else if (comment2.getSentiment().equals(Sentiment.NEUTRAL)) {
                        positiveNeutral++;
                    } else {
                        other++;
                    }
                } else if (comment1.getSentiment().equals(Sentiment.NEGATIVE)) {
                    if (comment2.getSentiment().equals(Sentiment.POSITIVE)) {
                        negativePositive++;
                    } else if (comment2.getSentiment().equals(Sentiment.NEGATIVE)) {
                        negativeNegative++;
                    }  else if (comment2.getSentiment().equals(Sentiment.NEUTRAL)) {
                        negativeNeutral++;
                    } else {
                        other++;
                    }
                } else if (comment1.getSentiment().equals(Sentiment.NEUTRAL)) {
                    if (comment2.getSentiment().equals(Sentiment.POSITIVE)) {
                        neutralPositive++;
                    } else if (comment2.getSentiment().equals(Sentiment.NEGATIVE)) {
                        neutralNegative++;
                    }  else if (comment2.getSentiment().equals(Sentiment.NEUTRAL)) {
                        neutralNeutral++;
                    } else {
                        other++;
                    }
                } else {
                    other++;
                }

                String commentEntry = newsArticle1.getArticleId() + "_"
                        + comment1.getIndex() + "," + comment1.getSentiment() + "," + comment2.getSentiment();
                commentEntryList.add(commentEntry);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to JAXB");
        }

        return commentEntryList;
    }

    private void writeToFiles() {
        String commentFile = "./corpus/analyzed/comments_kappa.csv";

        try (FileWriter fileWriter = new FileWriter(commentFile)) {
            fileWriter.write(aggregateComments.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


//positivePositive = 165
//positiveNegative = 4
//positiveNeutral = 46
//negativePositive = 19
//negativeNegative = 488
//negativeNeutral = 83
//neutralPositive = 13
//neutralNegative = 23
//neutralNeutral = 138
//other = 1
//Total comments =  980
//Po = 0.8071428571428572, Pe = 0.4088536026655561
//Agreement percentage = 0.8071428571428572
//kappa = 0.673757391186412