package com.isuru.crawler;

import edu.uci.ics.crawler4j.crawler.CrawlConfig;
import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.fetcher.PageFetcher;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtConfig;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtServer;

public class MainCrawler {
    public static void main(String[] args) throws Exception {
        String crawlStorageFolder = "./out";
        int numberOfCrawlers = 7;

        CrawlConfig config = new CrawlConfig();
        config.setCrawlStorageFolder(crawlStorageFolder);

        PageFetcher pageFetcher = new PageFetcher(config);
        RobotstxtConfig robotstxtConfig = new RobotstxtConfig();
        RobotstxtServer robotstxtServer = new RobotstxtServer(robotstxtConfig, pageFetcher);
        CrawlController controller = new CrawlController(config, pageFetcher, robotstxtServer);

        controller.addSeed("http://www.lankadeepa.lk/latest_news/%E0%B6%9A%E0%B7%90%E0%B6%B6%E0%B7%92%E0%B6%B1%E0%B6%A7%E0%B7%8A-%E0%B7%83%E0%B6%82%E0%B7%81%E0%B7%9D%E0%B6%B0%E0%B6%B1%E0%B6%BA%E0%B6%9A%E0%B7%8A-%E0%B7%80%E0%B7%99%E0%B6%B1%E0%B7%8A%E0%B6%B1%E0%B7%9A-%E0%B6%B1%E0%B7%91---%E0%B6%9A%E0%B7%90%E0%B6%B6%E0%B7%92%E0%B6%B1%E0%B6%A7%E0%B7%8A-%E0%B6%B4%E0%B7%8A%E2%80%8D%E0%B6%BB%E0%B6%9A%E0%B7%8F%E0%B7%81%E0%B6%9A-%E0%B6%BB%E0%B7%8F%E0%B6%A2%E0%B7%92%E0%B6%AD/1-512603");
        controller.addSeed("http://www.lankadeepa.lk/latest_news/%E0%B6%9A%E0%B7%94%E2%80%8D%E0%B6%AB%E0%B7%94-%E0%B6%9A%E0%B6%B1%E0%B7%8A%E0%B6%AF%E0%B7%9A-%E0%B6%85%E0%B6%AD%E0%B7%93%E0%B6%AD%E0%B6%BA--%E0%B7%80%E0%B6%BB%E0%B7%8A%E0%B6%AD%E0%B6%B8%E0%B7%8F%E0%B6%B1%E0%B6%BA-%E0%B7%83%E0%B7%84-%E0%B6%85%E0%B6%B1%E0%B7%8F%E0%B6%9C%E0%B6%AD%E0%B6%BA-%E0%B6%9C%E0%B7%90%E0%B6%B1-%E0%B6%87%E0%B6%B8%E0%B6%AD%E0%B7%92-%E0%B6%B4%E0%B7%8F%E0%B6%A8%E0%B6%BD%E0%B7%93-%E0%B6%9A%E0%B7%92%E0%B6%BA%E0%B6%BA%E0%B7%92/1-511879");
        controller.addSeed("http://www.lankadeepa.lk/archives");
        controller.addSeed("http://www.lankadeepa.lk/");
        controller.addSeed("http://www.lankadeepa.lk/politics/%E0%B6%B6%E0%B6%A9%E0%B7%94-%E0%B6%B4%E0%B7%92%E0%B6%A7%E0%B6%BB%E0%B6%A7-%E0%B6%BA%E0%B7%80%E0%B7%8F--%E0%B6%BD%E0%B7%9D%E0%B6%9A-%E0%B7%80%E0%B7%99%E0%B7%85%E0%B7%99%E0%B6%B3%E0%B6%B4%E0%B7%9C%E0%B7%85-%E0%B6%85%E0%B6%BD%E0%B7%8A%E0%B6%BD%E0%B7%8F-%E0%B6%9C%E0%B7%90%E0%B6%B1%E0%B7%93%E0%B6%B8%E0%B6%A7-%E0%B7%81%E0%B7%8A%E2%80%8D%E0%B6%BB%E0%B7%93-%E0%B6%BD%E0%B6%82%E0%B6%9A%E0%B7%8F%E0%B7%80-%E0%B7%83%E0%B7%96%E0%B6%AF%E0%B7%8F%E0%B6%B1%E0%B6%B8%E0%B7%8A/13-512530");
        controller.addSeed("http://www.lankadeepa.lk/politics/13");
        controller.addSeed("http://www.lankadeepa.lk/latest_news/%E0%B7%80%E0%B7%92%E0%B6%AF%E0%B7%9A%E0%B7%81-%E0%B7%80%E0%B7%92%E0%B6%B1%E0%B7%92%E0%B6%B8%E0%B6%BA-%E0%B6%B4%E0%B7%8F%E0%B6%BD%E0%B6%B1-%E0%B6%B4%E0%B6%B1%E0%B6%AD-%E0%B7%83%E0%B6%82%E0%B7%81%E0%B7%9D%E0%B6%B0%E0%B6%B1%E0%B6%BA-%E0%B6%B1%E0%B6%BB%E0%B7%92%E0%B6%B1%E0%B7%8A%E0%B6%A7-%E0%B6%9A%E0%B7%94%E0%B6%9A%E0%B7%94%E0%B7%85%E0%B6%B1%E0%B7%8A-%E0%B6%B6%E0%B7%8F%E0%B6%BB%E0%B6%AF%E0%B7%93%E0%B6%B8%E0%B6%9A%E0%B7%8A/1-512614");

        controller.start(CommentCrawler.class, numberOfCrawlers);
    }
}
