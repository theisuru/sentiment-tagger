package com.isuru.tagger;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    private static ArticleReader articleReader;

    public static void main(String[] args) {
        init();
        SpringApplication.run(Application.class, args);
    }

    private static void init() {
        articleReader = new ArticleReader("./corpus");
    }

    public static ArticleReader getArticleReader() {
        return articleReader;
    }

}
