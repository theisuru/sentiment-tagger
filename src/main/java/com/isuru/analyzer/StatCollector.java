package com.isuru.analyzer;

import com.isuru.bean.Comment;
import com.isuru.bean.NewsArticle;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

import static java.util.Objects.requireNonNull;

/**
 * Extract comments from NewsArticle.
 */
public class StatCollector {
    private static final Logger logger = Logger.getLogger("StatCollector");

    private static Unmarshaller jaxbUnmarshaller;
    private static Set<String> articleWordSet = new HashSet<>();
    private static Set<String> commentWordSet = new HashSet<>();

    private static long noOfArticles = 0;
    private static long noOfComments = 0;
    private static long totalArticleWords = 0;
    private static long totalCommentWords = 0;

    private static List<Integer> articleCommentCountList = new ArrayList<>();
    private static Map<String, Integer> dateCommentCountMap = new HashMap<>();

    public static void main(String[] args) throws Exception {
        File folder = new File("./corpus/raw_data");
        File[] listOfFiles = folder.listFiles();
        requireNonNull(listOfFiles, "No data files to begin with");

        JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
        jaxbUnmarshaller = jaxbContext.createUnmarshaller();

        for (int i = 0; i < 1000; i++) {
            articleCommentCountList.add(0);
        }

        for (File file : listOfFiles) {
            if (file.isFile()) {
                logger.info("File: " + file.getName());
                count(file);
            } else if (file.isDirectory()) {
                logger.info("Directory: " + file.getName());
            }
        }
        writeToFiles();
    }

    private static void count(File file) throws Exception {
        NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);
        String s;
        int commentsPerFile = 0;
        String commentDate;

        ++noOfArticles;
        s = newsArticle.getBody().replaceAll("[\\s*(\\p{Punct})+\\s*]", " ");
        s = s.replaceAll("\\s+", " ");
        String[] articleWords = s.split(" ");
        articleWordSet.addAll(new HashSet<>(Arrays.asList(articleWords)));
        totalArticleWords += articleWords.length;

        for (Comment comment : newsArticle.getComments()) {
            ++noOfComments;
            s = comment.getPhrase().replaceAll("[\\s*(\\p{Punct})+\\s*]", " ");
            s = s.replaceAll("\\s+", " ");
            String[] commentWords = s.split(" ");
            commentWordSet.addAll(new HashSet<>(Arrays.asList(commentWords)));
            totalCommentWords += commentWords.length;

            ++commentsPerFile;
            commentDate = comment.getDate().substring(0, 10);
            if (!commentDate.equals("") && !commentDate.equals(" ")) {
                if (dateCommentCountMap.containsKey(commentDate)) {
                    dateCommentCountMap.put(commentDate, dateCommentCountMap.get(commentDate) + 1);
                } else {
                    dateCommentCountMap.put(commentDate, 1);
                }
            }
        }
        articleCommentCountList.set(commentsPerFile, articleCommentCountList.get(commentsPerFile) + 1);
    }

    private static void writeToFiles() {
        String basicStat = "./results/basic_stat.txt";
        String commentCount = "./results/comments_count.csv";
        String commentDates = "./results/comment_dates_count.csv";

        try (FileWriter fileWriter = new FileWriter(basicStat)) {
            fileWriter.write("number of articles: " + noOfArticles + "\n");
            fileWriter.write("number of comments: " + noOfComments + "\n");
            fileWriter.write("total number of article words: " + totalArticleWords + "\n");
            fileWriter.write("total number of comment words: " + totalCommentWords + "\n");
            fileWriter.write("number of unique words in comments: " + articleWordSet.size() + "\n");
            fileWriter.write("number of unique words in comments: " + commentWordSet.size() + "\n\n");
            fileWriter.write("average article words: " + totalArticleWords / noOfArticles + "\n");
            fileWriter.write("average comment words: " + totalCommentWords / noOfComments + "\n");
            fileWriter.write("average comments per article: " + noOfComments / noOfArticles + "\n");
            fileWriter.flush();
        } catch (IOException e) {
            logger.severe("Failed to write to the output file: " + e.toString());
        }

        try (FileWriter fileWriter = new FileWriter(commentCount)) {
            fileWriter.write("noOfComments, articleCount\n");
            for (int i = 0; i < articleCommentCountList.size(); i++) {
                fileWriter.write(i + "," + articleCommentCountList.get(i) + "\n");
            }
            fileWriter.flush();
        } catch (IOException e) {
            logger.severe("Failed to write to the output file: " + e.toString());
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
            logger.severe("Failed to write to the output file: " + e.toString());
        }
    }
}
