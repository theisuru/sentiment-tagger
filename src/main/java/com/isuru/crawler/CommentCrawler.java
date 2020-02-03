package com.isuru.crawler;

import com.isuru.bean.NewsArticle;
import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;

import java.util.Set;
import java.util.regex.Pattern;


public class CommentCrawler extends WebCrawler {
    private final static Pattern FILTERS = Pattern.compile(".*(\\.(css|js|gif|jpg"
            + "|png|mp3|mp3|zip|gz))$");

    @Override
    public boolean shouldVisit(Page referringPage, WebURL url) {
        String href = url.getURL().toLowerCase();
        return !FILTERS.matcher(href).matches()
                && href.startsWith("http://www.lankadeepa.lk/");
    }

    @Override
    public void visit(Page page) {
        String url = page.getWebURL().getURL();
        logger.info("URL: " + url);
        String[] urlParts = url.split("/");
        String fileName = urlParts[urlParts.length - 1];

        if (page.getParseData() instanceof HtmlParseData) {
            HtmlParseData htmlParseData = (HtmlParseData) page.getParseData();
            String text = htmlParseData.getText();
            String html = htmlParseData.getHtml();
            Set<WebURL> links = htmlParseData.getOutgoingUrls();

            logger.info("Text length: " + text.length());
            logger.info("Html length: " + html.length());
            logger.info("Number of outgoing links: " + links.size());

            int beginingIndex = 0;
            int noOfComments = 0;
            if (html.contains("අදහස් (")) {
                beginingIndex = html.indexOf("අදහස් (");
                noOfComments = Integer.parseInt(html.substring(beginingIndex + 7, beginingIndex + 8));
            }

            if (noOfComments > 0) {
                NewsArticle article = CommentExtractor.extractNews(html, url);
                CommentExtractor.saveXML(fileName, article);
//            	 CommentExtractor.saveJson(fileName, article);
                CommentExtractor.saveTxt(fileName, article);
                CommentExtractor.saveAnn(fileName, article);
                logger.info(article.toString());
            }
        }
    }
}
