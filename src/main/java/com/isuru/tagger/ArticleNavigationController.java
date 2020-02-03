package com.isuru.tagger;

import com.isuru.bean.Comment;
import com.isuru.bean.NewsArticle;
import com.isuru.bean.Sentiment;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.xml.bind.JAXBException;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

@Controller
public class ArticleNavigationController {
    private static final Logger logger = Logger.getLogger("ArticleNavigationController");

    @RequestMapping(value = "/newsarticle/{articleIndex}", method = RequestMethod.GET)
    public String displayArticle(@PathVariable(value = "articleIndex") Integer articleIndex,
                                 Model model) {
        logger.info("get request for news article, index : " + articleIndex);
        NewsArticle newsArticle = null;
        try {
            ArticleReader articleReader = Application.getArticleReader();
            newsArticle = articleReader.getNewsArticle(articleIndex);
        } catch (JAXBException e) {
            e.printStackTrace();
            logger.warning("something went very wrong");
        }

        model.addAttribute("currentArticle", "/newsarticle/" + (articleIndex));
        model.addAttribute("nextArticle", "/newsarticle/" + (articleIndex + 1));
        model.addAttribute("prevArticle", "/newsarticle/" + (articleIndex - 1));
        model.addAttribute("newsArticle", newsArticle);

        model.addAttribute("sentimentValues", Arrays.asList(Sentiment.values()));
        return "navigateArticle";
    }

    @RequestMapping(value = "/processForm", method = RequestMethod.POST)
    public String save(@ModelAttribute("newsArticle") NewsArticle newsArticle,
                       @ModelAttribute(value = "comments") HashMap<String, String> comments,
                       @ModelAttribute(value = "currentArticle") String currentArticle,
                       Model model) throws ClassNotFoundException, IOException,
            InterruptedException {


        ArticleReader articleReader = Application.getArticleReader();
        NewsArticle originalNewsArticle = null;
        try {
            originalNewsArticle = articleReader.getNewsArticle(newsArticle.getArticleId());
        } catch (JAXBException e) {
            e.printStackTrace();
        }

        List<Comment> currentComments = newsArticle.getComments();
        if (originalNewsArticle != null && !currentComments.isEmpty()) {
            for (int i = 0; i < originalNewsArticle.getComments().size(); i++) {
                Comment originalComment = originalNewsArticle.getComments().get(i);
                Comment currentComment = currentComments.get(i);
                if (currentComment.getIndex() == originalComment.getIndex()) {
                    if (currentComment.getSentiment() != null) {
                        originalComment.setSentiment(currentComment.getSentiment());
                    }
                } else {
                    logger.info("Original index does not match with current index.");
                }
            }
        } else {
            logger.info("Either no comments in the article or something worst");
        }

        articleReader.saveFile(originalNewsArticle, articleReader.getArticleIndex(newsArticle.getArticleId()), currentComments.size());

        return "redirect:" + "/newsarticle/" + (articleReader.getArticleIndex(newsArticle.getArticleId()) + 1);
    }

}
