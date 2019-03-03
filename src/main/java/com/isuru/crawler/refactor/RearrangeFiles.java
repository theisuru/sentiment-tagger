package com.isuru.crawler.refactor;

import com.isuru.bean.NewsArticle;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * From @OldNewsArticle format to @NewsArticle.
 */
public class RearrangeFiles {
    private static final Logger logger = Logger.getLogger("RearrangeFiles");
    
    public static void main(String[] args) {
        File folder = new File("./corpus/old2");
        File[] listOfFiles = folder.listFiles();
        RearrangeFiles rearrangeFiles = new RearrangeFiles();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                logger.info("File " + listOfFiles[i].getName());
                rearrangeFiles.convertToNewFormat2(listOfFiles[i]);
            } else if (listOfFiles[i].isDirectory()) {
                logger.info("Directory " + listOfFiles[i].getName());
            }
        }
        /*File file = new File("./corpus/old2/1149.txt");//121526
        new RearrangeFiles().convertToNewFormat2(file);*/
    }

    private static void convertToNewFormat(File file) {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(OldNewsArticle.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();

            OldNewsArticle oldNewsArticle = (OldNewsArticle) jaxbUnmarshaller.unmarshal(file);
            NewsArticle newsArticle = oldNewsArticle.getNewerVersion();

            jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Marshaller jaxbMarshaller = jaxbContext.createMarshaller();
            jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);

            String fileName = file.getName().substring(0, file.getName().length() - 4);
            jaxbMarshaller.marshal(newsArticle, new File("./corpus/old2/" + fileName + ".txt"));
            new File("./corpus/ann/" + fileName + ".ann").createNewFile();

        } catch (JAXBException | IOException e) {
            e.printStackTrace();
        }
    }

    private void convertToNewFormat2(File file) {
        try {
            String fileName = file.getName().substring(0, file.getName().length() - 4);

            JAXBContext jaxbContext = JAXBContext.newInstance(OldNewsArticle2.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();

            OldNewsArticle2 oldNewsArticle2 = (OldNewsArticle2) jaxbUnmarshaller.unmarshal(file);
            NewsArticle newsArticle = oldNewsArticle2.getNewerVersion(Long.parseLong(fileName));

            jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            Marshaller jaxbMarshaller = jaxbContext.createMarshaller();
            jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);

            jaxbMarshaller.marshal(newsArticle, new File("./corpus/new/" + fileName + ".txt"));
//            new File("./corpus/ann/" + fileName + ".ann").createNewFile();

        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }
}

