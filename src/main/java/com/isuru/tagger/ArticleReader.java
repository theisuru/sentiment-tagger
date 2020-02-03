package com.isuru.tagger;

import com.isuru.bean.NewsArticle;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.logging.Logger;

/**
 * Reads xml @NewsArticle from the disk..
 */
public class ArticleReader {
    private static final Logger logger = Logger.getLogger("ArticleReader");

    private String rootPath;
    private List<File> articleFileList;
    private Map<String, File> articleFileMap;

    private Unmarshaller jaxbUnmarshaller;
    private Marshaller jaxbMarshaller;

    public ArticleReader(String rootPath) {
        this.rootPath = rootPath;
        init();
        fillArticleList(rootPath + "/raw_data");
    }

    private void init() {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(NewsArticle.class);
            jaxbUnmarshaller = jaxbContext.createUnmarshaller();

            jaxbMarshaller = jaxbContext.createMarshaller();
            jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }

    private List<File> getFileArray(String folderPath) {
        File folder = new File(folderPath);
        File[] fileArray = folder.listFiles();
        List<File> fileList = new ArrayList<>(fileArray.length);

        for (int i = 0; i < fileArray.length; i++) {
            if (fileArray[i].isFile()) {
                fileList.add(fileArray[i]);
            } else if (fileArray[i].isDirectory()) {
                logger.info("Directory " + fileArray[i].getName());
            }
        }
        return fileList;
    }

    private void fillArticleList(String folderPath) {
        File folder = new File(folderPath);
        File[] fileArray = folder.listFiles();
        Arrays.sort(fileArray);
        articleFileList = new ArrayList<>(fileArray.length);
        articleFileMap = new HashMap<>(fileArray.length);

        for (int i = 0; i < fileArray.length; i++) {
            if (fileArray[i].isFile()) {
                articleFileList.add(fileArray[i]);
                String fileName = fileArray[i].getName();
                articleFileMap.put(fileName.substring(0, (fileName.length() - 4)), fileArray[i]);
            } else if (fileArray[i].isDirectory()) {
                logger.info("Directory " + fileArray[i].getName());
            }
        }
    }

    //get from index
    public NewsArticle getNewsArticle(int articleIndex) throws JAXBException {
        return (NewsArticle) jaxbUnmarshaller.unmarshal(articleFileList.get(articleIndex));
    }

    //get from article name prefix (id)
    public NewsArticle getNewsArticle(long articleId) throws JAXBException {
        return (NewsArticle) jaxbUnmarshaller.unmarshal(articleFileMap.get(String.valueOf(articleId)));
    }

    public List<NewsArticle> getNewsArticles(int first, int last) throws JAXBException {
        List<NewsArticle> articleList = new ArrayList<>(last - first);
        for (int i = first; i < last; i++) {
            articleList.add((NewsArticle) jaxbUnmarshaller.unmarshal(articleFileList.get(i)));
        }
        return articleList;
    }

    public int getArticleIndex(long articleId) {
        return articleFileList.indexOf(articleFileMap.get(String.valueOf(articleId)));
    }

    public List<File> getArticleFileList() {
        return articleFileList;
    }

    public void setArticleFileList(List<File> articleFileList) {
        this.articleFileList = articleFileList;
    }

    public Map<String, File> getArticleFileMap() {
        return articleFileMap;
    }

    public void setArticleFileMap(Map<String, File> articleFileMap) {
        this.articleFileMap = articleFileMap;
    }

    public Unmarshaller getJaxbUnmarshaller() {
        return jaxbUnmarshaller;
    }

    public void setJaxbUnmarshaller(Unmarshaller jaxbUnmarshaller) {
        this.jaxbUnmarshaller = jaxbUnmarshaller;
    }

    public void saveFile(NewsArticle article) {
        File output = new File(rootPath + "/tagged/" + article.getArticleId() + ".txt");
        try {
            jaxbMarshaller.marshal(article, output);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void saveFile(NewsArticle article, int index, int noOfComments) {
        saveFile(article);
        String output = rootPath + "/index.txt";
        try (FileWriter writer = new FileWriter(output, true)) {
            writer.write(index + "\t\t" + article.getArticleId() + "\t\t" + noOfComments + "\t\t" + new Date() + "\n");
            writer.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
