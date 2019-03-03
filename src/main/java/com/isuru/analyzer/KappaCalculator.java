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

    private static List<String> fileNames = Arrays.asList(/*"101446.txt",*/ /*"102878.txt",*/ /*"102999.txt",*/
            "103166.txt", "103261.txt"/*, "103346.txt"*/, "103448.txt");

    List<String> commentEntryList = new ArrayList<>();

    private static int positivePositive = 0;
    private static int positiveNegative = 0;
    private static int positiveOther = 0;
    private static int negativePositive = 0;
    private static int negativeNegative = 0;
    private static int negativeOther = 0;
    private static int otherPositive = 0;
    private static int otherNegative = 0;
    private static int otherOther = 0;

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
        System.out.println("positiveOther = " + positiveOther);
        System.out.println("negativePositive = " + negativePositive);
        System.out.println("negativeNegative = " + negativeNegative);
        System.out.println("negativeOther = " + negativeOther);
        System.out.println("otherPositive = " + otherPositive);
        System.out.println("otherNegative = " + otherNegative);
        System.out.println("otherOther = " + otherOther);

        int agree = positivePositive + negativeNegative + otherOther;
        int all = positivePositive + positiveNegative + positiveOther +
                negativePositive + negativeNegative + negativeOther +
                otherPositive + otherNegative + otherOther;

        int rater1Positive = positivePositive + positiveNegative + positiveOther;
        int rater1Negative = negativePositive + negativeNegative + negativeOther;
        int rater1Other = otherPositive + otherNegative + otherOther;
        int rater2Positive = positivePositive + negativePositive + otherPositive;
        int rater2Negative = positiveNegative + negativeNegative + otherNegative;
        int rater2Other = positiveOther + negativeOther +otherOther;

        double po = agree * 1.0 / all;
        double pe = 1.0 / all /all * (rater1Positive * rater2Positive + rater1Negative * rater2Negative + rater1Other * rater2Other);
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
                    } else {
                        positiveOther++;
                    }
                } else if (comment1.getSentiment().equals(Sentiment.NEGATIVE)) {
                    if (comment2.getSentiment().equals(Sentiment.POSITIVE)) {
                        negativePositive++;
                    } else if (comment2.getSentiment().equals(Sentiment.NEGATIVE)) {
                        negativeNegative++;
                    } else {
                        negativeOther++;
                    }
                } else {
                    if (comment2.getSentiment().equals(Sentiment.POSITIVE)) {
                        otherPositive++;
                    } else if (comment2.getSentiment().equals(Sentiment.NEGATIVE)) {
                        otherNegative++;
                    } else {
                        otherOther++;
                    }
                }

                String commentEntry = newsArticle1.getArticleId() + "_"
                        + comment1.getIndex() + "," + comment1.getSentiment() + "," + comment2.getSentiment();
                commentEntryList.add(commentEntry);
            }
        } catch (Exception e) {
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

    /*public static void main(String[] args) {
        String s = "මේවා ආණ්ඩුවේ වැරදි නොව  නිලධාරීන්ගේ නොහොබිනා වැඩ  මෙවන් " +
                "නිලධාරීන් සෑම  කාර්ය්\u200Dයාලයකම ඉන්නවා මොවුන්  අතයටින් මුදල්  දෙනතුරු මේවායේ වැඩ නොකෙරෙයි";

        String s1 = s.replaceAll("  ", " ");

        StringBuilder s2 = new StringBuilder();
        s2.append(s.replace(".", " ")
                .replace(",", " ")
                .replace(":", " ")
                .replace("!", " ")
                .replace(";", " ")
                .replace("\"", " ")
                .replace("'", " ")
                .replaceAll("  ", " ")
                .replaceAll("  ", " ")
                .replaceAll("\\s+", " "));

        String s3 = s.replace("\u00A0", " ");

        String s4 = s.replaceAll("(\\h+)"," ");

        System.out.println(s);
        System.out.println(s1);
        System.out.println(s2.toString());
        System.out.println(s3);
        System.out.println(s4);
    }*/
}
