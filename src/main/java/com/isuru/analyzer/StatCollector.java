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
import java.util.*;
import java.util.logging.Logger;

/**
 * Extract comments from NewsArticle.
 */
public class StatCollector {
    private static final Logger logger = Logger.getLogger("StatCollector");

    private static final String SEPARATOR = ";";
    private static final String NEW_LINE = "\n";

    private static Unmarshaller jaxbUnmarshaller;

    private static long noOfArticles = 0;
    private static long noOfComments = 0;
    private static long totalArticleWords = 0;
    private static long totalCommentWords = 0;

    private static List<Integer> articleCommentCountList = new ArrayList<>();
    private static Map<String, Integer> dateCommentCountMap = new HashMap<>();

    private static long avgArticleWordCount = 0;
    private static long avgCommentWordCount = 0;
    private static long avgNoOfCommentsPerArticle = 0;

    public static void main(String[] args) throws Exception {
        File folder = new File("./corpus/new");
        File[] listOfFiles = folder.listFiles();
        StatCollector statCollector = new StatCollector();
        JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
        jaxbUnmarshaller = jaxbContext.createUnmarshaller();

        for (int i = 0; i < 1000; i++) {
            articleCommentCountList.add(0);
        }

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                logger.info("File " + listOfFiles[i].getName());
                statCollector.count(listOfFiles[i]);
            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }
        }
        statCollector.writeToFiles();
    }

    private void count(File file) throws Exception {
        NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);
        String s;
        int commentsPerFile = 0;
        String commentDate;

        ++noOfArticles;
        s = newsArticle.getBody().replaceAll("[\\s*(\\p{Punct})+\\s*]", " ");
        s = s.replaceAll("\\s+", " ");
        totalArticleWords += s.split(" ").length;

        for (Comment comment : newsArticle.getComments()) {
            ++noOfComments;
            s = comment.getPhrase().replaceAll("[\\s*(\\p{Punct})+\\s*]", " ");
            s = s.replaceAll("\\s+", " ");
            totalCommentWords += s.split(" ").length;

            ++commentsPerFile;
            commentDate = comment.getDate().substring(0, 10);
            if(commentDate != null && !commentDate.equals("") && !commentDate.equals(" ")) {
                if (dateCommentCountMap.containsKey(commentDate)) {
                    dateCommentCountMap.put(commentDate, dateCommentCountMap.get(commentDate) + 1);
                } else {
                    dateCommentCountMap.put(commentDate, 1);
                }
            }
        }
        articleCommentCountList.set(commentsPerFile, articleCommentCountList.get(commentsPerFile) + 1);
    }

    private void writeToFiles() {
        String basicStat = "./corpus/analyzed/statBasic.csv";
        String commentCount = "./corpus/analyzed/statCommentCount.csv";
        String commentDates = "./corpus/analyzed/statCommentDates.csv";

        try (FileWriter fileWriter = new FileWriter(basicStat)) {
            fileWriter.write("# articles: " + noOfArticles + "\n");
            fileWriter.write("# comments: " + noOfComments + "\n");
            fileWriter.write("total # article words: " + totalArticleWords + "\n");
            fileWriter.write("total # comment words: " + totalCommentWords + "\n");
            fileWriter.write("average article words: " + totalArticleWords / noOfArticles + "\n");
            fileWriter.write("average comment words: " + totalCommentWords / noOfComments + "\n");
            fileWriter.write("average comments per article: " + noOfComments / noOfArticles + "\n");
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter fileWriter = new FileWriter(commentCount)) {
            fileWriter.write("noOfComments, articleCount\n");
            for (int i = 0; i < articleCommentCountList.size(); i++) {
                fileWriter.write(i + "," + articleCommentCountList.get(i) + "\n");
            }
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter fileWriter = new FileWriter(commentDates)) {
            fileWriter.write("commentDate, commentCount\n");
            Object[] keys = dateCommentCountMap.keySet().toArray();
            Arrays.sort(keys);
            for (Object key : keys) {
                fileWriter.write(key + "," + dateCommentCountMap.get(key) + "\n");
            }
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
