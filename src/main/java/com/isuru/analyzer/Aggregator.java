package com.isuru.analyzer;

import com.isuru.bean.Comment;
import com.isuru.bean.NewsArticle;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Aggregate news and comments.
 */
public class Aggregator {
    private static final Logger logger = Logger.getLogger("Aggregator");
    
    private static final String SEPARATOR = ";";
    private static final String NEW_LINE = "\n";
    private StringBuilder aggregateComments = new StringBuilder();
    private StringBuilder aggregateNews = new StringBuilder();

    public static void main(String[] args) {
        File folder = new File("./corpus/new");
        File[] listOfFiles = folder.listFiles();
        Aggregator aggregator = new Aggregator();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                logger.info("File " + listOfFiles[i].getName());
//                aggregator.aggregate(listOfFiles[i]);
                aggregator.aggregateWithFileName(listOfFiles[i]);
            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }

            //stop prematurely
            if (i == 3000) {
                break;
            }
        }
        aggregator.writeToFiles();
    }

    private void aggregate(File file) {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
            NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);

            aggregateNews
                    .append(newsArticle.getBody()
                            .replace(SEPARATOR, ",")
                            .replace(NEW_LINE, ". "))
                    .append(NEW_LINE);
            for (Comment comment : newsArticle.getComments()) {
                aggregateComments.append(comment.getPhrase()
                        .replace(SEPARATOR, ",")
                        .replace(NEW_LINE, ". "))
                        .append(NEW_LINE);
            }
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }

    private void aggregateWithFileName(File file) {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
            NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);
            String fileName = file.getName().split(".txt")[0];

            aggregateNews
                    .append(fileName)
                    .append(SEPARATOR)
                    .append(
                            newsArticle.getBody()
                                    .replace(SEPARATOR, ",")
                                    .replace(NEW_LINE, ". "))
                    .append(NEW_LINE);
            for (Comment comment : newsArticle.getComments()) {
                aggregateComments
                        .append(fileName)
                        .append(SEPARATOR)
                        .append(comment.getPhrase()
                        .replace(SEPARATOR, ",")
                        .replace(NEW_LINE, ". "))
                        .append(NEW_LINE);
            }
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }

    private void writeToFiles() {
        String commentFile = "./corpus/analyzed/comments.csv";
        String newsFile = "./corpus/analyzed/news.csv";

        try (FileWriter fileWriter = new FileWriter(newsFile)) {
            fileWriter.write(aggregateNews.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter fileWriter = new FileWriter(commentFile)) {
            fileWriter.write(aggregateComments.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
