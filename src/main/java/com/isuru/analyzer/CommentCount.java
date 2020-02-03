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
import java.util.logging.Logger;

/**
 * Extract comments from NewsArticle.
 */
public class CommentCount {
    private static final Logger logger = Logger.getLogger("Aggregator");
    private static int positiveComments = 0;
    private static int negativeComments = 0;
    private static int undefinedComments = 0;
    private static int neutralComments = 0;
    private static int conflictComments = 0;
    private static int otherComments = 0;
    private static int totalComments = 0;
    private static int totalFiles = 0;

    public static void main(String[] args) {
        File folder = new File("./corpus/tagged");
        File[] listOfFiles = folder.listFiles();
        CommentCount extractor = new CommentCount();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                totalFiles++;
                logger.info("File " + listOfFiles[i].getName());
                extractor.aggregateWithFileName(listOfFiles[i]);
            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }
        }
        System.out.println("Finished processing");
        System.out.println("Total comments: " + totalComments);
        System.out.println("Positive comments: " + positiveComments);
        System.out.println("Negative comments: " + negativeComments);
        System.out.println("Neutral comments: " + neutralComments);
        System.out.println("Conflict comments: " + conflictComments);
        System.out.println("Undefined comments: " + undefinedComments);
        System.out.println("Other comments: " + otherComments);

        System.out.println("Total files: " + totalFiles);
        System.out.println("Comments to pay for: " + (positiveComments + negativeComments + otherComments));
        System.out.println("Total payment : " + 0.5 * (positiveComments + negativeComments + otherComments));
    }

    private void aggregateWithFileName(File file) {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
            NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);

            for (Comment comment : newsArticle.getComments()) {
                totalComments++;
                if (comment.getSentiment().equals(Sentiment.POSITIVE)) {
                    positiveComments++;
                } else if (comment.getSentiment().equals(Sentiment.NEGATIVE)) {
                    negativeComments++;
                } else if (comment.getSentiment().equals(Sentiment.UNDEFINED)) {
                    undefinedComments++;
                } else if (comment.getSentiment().equals(Sentiment.NEUTRAL)) {
                    neutralComments++;
                } else if (comment.getSentiment().equals(Sentiment.CONFLICT)) {
                    conflictComments++;
                } else {
                    otherComments++;
                }
            }
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }
}
