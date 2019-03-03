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
public class CommentExtractor {
    private static final Logger logger = Logger.getLogger("Aggregator");

    private static final String SEPARATOR = ";";
    private static final String NEW_LINE = "\n";
    private StringBuilder aggregateComments = new StringBuilder().append("docid;comment;label\n");

    public static void main(String[] args) {
        File folder = new File("./corpus/tagged");
        File[] listOfFiles = folder.listFiles();
        CommentExtractor extractor = new CommentExtractor();
        long totalCommentsAdded = 0;

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                logger.info("File " + listOfFiles[i].getName());
                totalCommentsAdded += extractor.aggregateWithFileName(listOfFiles[i]);
            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }
        }
        logger.info("Finished processing, writing to file comments_tagged.csv, #comments = " + totalCommentsAdded);
        extractor.writeToFiles();
    }

    private int aggregateWithFileName(File file) {
        int usefulComments = 0;
        int uselessComments = 0;
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
            NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);
            String fileName = file.getName().split(".txt")[0];

            for (Comment comment : newsArticle.getComments()) {
                if (comment.getSentiment().equals(Sentiment.POSITIVE) ||
                        comment.getSentiment().equals(Sentiment.NEGATIVE)) {
                    usefulComments++;
                    aggregateComments
                            .append(fileName)
                            .append(SEPARATOR)
                            .append(comment.getPhrase()
                                    .replace(SEPARATOR, ",")
                                    .replace(NEW_LINE, ". ")
//                                    .replaceAll("[\\s*(\\p{Punct})+\\s*]", " ")
                                    .replace(".", " ")
                                    .replace(",", " ")
                                    .replace(":", " ")
                                    .replace("-", " ")
                                    .replace("!", " ")
                                    .replace(";", " ")
                                    .replace("[", " ")
                                    .replace("]", " ")
                                    .replace("(", " ")
                                    .replace(")", " ")
                                    .replace("\\", " ")
                                    .replace("/", " ")
                                    .replace("\"", " ")
                                    .replace("'", " ")
                                    .replaceAll("(\\h+)"," ")
                                    .replaceAll("\\s+", " ")
                                    .trim())
                            .append(SEPARATOR)
                            .append(comment.getSentiment().toString())
                            .append(NEW_LINE);
                } else {
                    uselessComments++;
                }
            }
        } catch (JAXBException e) {
            e.printStackTrace();
        }
        logger.info("Processed file " + file.getName() + " : " +
                "with comments " + usefulComments + ", " + uselessComments);
        return usefulComments;
    }

    private void writeToFiles() {
        String commentFile = "./corpus/analyzed/comments_tagged_remove_all_punc_keep_question.csv";

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
