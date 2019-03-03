package com.isuru.analyzer;

import com.isuru.bean.NewsArticle;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Rearranger file for better management support
 */
public class Rearranger {
    private static final Logger logger = Logger.getLogger("Rearranger");
    private static final int SEPARATE_COUNT = 1000;
    private static final List<Integer> commentCounter = new ArrayList<>(12);

    private Unmarshaller jaxbUnmarshaller;
    private Marshaller jaxbMarshaller;

    public Rearranger() {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            jaxbUnmarshaller = jaxbContext.createUnmarshaller();

            jaxbMarshaller = jaxbContext.createMarshaller();
            jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
        } catch (JAXBException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < 12; i++) {
            commentCounter.add(0);
        }
    }

    public static void main(String[] args) throws Exception {
        File originalFolder = new File("./corpus/raw_data");
        File[] listOfFiles = originalFolder.listFiles();
        Rearranger rearranger = new Rearranger();

        int folderCounter = 0;
        String destinationFolder = null;
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                if (i % SEPARATE_COUNT == 0) {
                    ++folderCounter;
                    destinationFolder = "./corpus/rearrange/1000_articles/" + folderCounter;
                    boolean success = (new File(destinationFolder)).mkdir();
                    if (success) {
                        logger.info("Created directory " + destinationFolder);
                    } else {
                        logger.info("Directory already exists " + destinationFolder);
                    }
                    destinationFolder = destinationFolder + "/";

                    StringBuilder stringBuilder = new StringBuilder();
                    for (int j = 0 ; j < commentCounter.size(); j++) {
                        stringBuilder.append((j * 5) + ":" + commentCounter.get(j) + ", ");
                    }
                    logger.info(stringBuilder.toString());
                }
                rearranger.saveFile(listOfFiles[i], destinationFolder);

            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }
        }
    }

    private void saveFile(File file, String folder) throws Exception {
        NewsArticle newsArticle = (NewsArticle) jaxbUnmarshaller.unmarshal(file);
        File destinationFile = new File(folder + file.getName());
        jaxbMarshaller.marshal(newsArticle, destinationFile);

        String folder2 = "./corpus/rearrange/no_of_comments/";
        int noOfComments = newsArticle.getComments().size();
        if (noOfComments == 0) {
            File destinationFile2 = new File(folder2 + "0/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(0);
            commentCounter.set(0, ++commentCount);
        } else if (noOfComments <= 5) {
            File destinationFile2 = new File(folder2 + "5/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(1);
            commentCounter.set(1, ++commentCount);
        } else if (noOfComments <= 10) {
            File destinationFile2 = new File(folder2 + "10/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(2);
            commentCounter.set(2, ++commentCount);
        } else if (noOfComments <= 15) {
            File destinationFile2 = new File(folder2 + "15/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(3);
            commentCounter.set(3, ++commentCount);
        } else if (noOfComments <= 20) {
            File destinationFile2 = new File(folder2 + "20/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(4);
            commentCounter.set(4, ++commentCount);
        } else if (noOfComments <= 25) {
            File destinationFile2 = new File(folder2 + "25/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(5);
            commentCounter.set(5, ++commentCount);
        } else if (noOfComments <= 30) {
            File destinationFile2 = new File(folder2 + "30/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(6);
            commentCounter.set(6, ++commentCount);
        } else if (noOfComments <= 35) {
            File destinationFile2 = new File(folder2 + "35/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(7);
            commentCounter.set(7, ++commentCount);
        } else if (noOfComments <= 40) {
            File destinationFile2 = new File(folder2 + "40/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(8);
            commentCounter.set(8, ++commentCount);
        } else if (noOfComments <= 45) {
            File destinationFile2 = new File(folder2 + "45/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(9);
            commentCounter.set(9, ++commentCount);
        } else if (noOfComments <= 50) {
            File destinationFile2 = new File(folder2 + "50/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(10);
            commentCounter.set(10, ++commentCount);
        } else {
            File destinationFile2 = new File(folder2 + "more/" + file.getName());
            jaxbMarshaller.marshal(newsArticle, destinationFile2);
            int commentCount = commentCounter.get(11);
            commentCounter.set(11, ++commentCount);
        }
    }
}
