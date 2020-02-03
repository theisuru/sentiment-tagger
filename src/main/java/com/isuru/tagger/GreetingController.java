package com.isuru.tagger;

import com.isuru.bean.Comment;
import com.isuru.bean.NewsArticle;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.ArrayList;
import java.util.List;

@Controller
public class GreetingController {

    @RequestMapping("/help")
    public String help(Model model) {
        return "help";
    }

    @RequestMapping("/greeting")
    public String greeting(@RequestParam(value="name", required=false, defaultValue="World") String name, Model model) {
        model.addAttribute("name", name);
        return "greeting";
    }

    @RequestMapping("/article")
    public String displayArticle(@RequestParam(value="name", required=false, defaultValue="World") String name, Model model) {
        model.addAttribute("author", "The ugly Writer");
        model.addAttribute("articleId", "1254698");
        model.addAttribute("date", "2017-10-03");
        model.addAttribute("title", "This is the news article title");
        model.addAttribute("body", "This is the news article body. Ideally this should contain some text.");
        model.addAttribute("url", "http://www.isuru.com/articleNumber/01");

        List<Comment> comments = new ArrayList<>();
        comments.add(new Comment("Isuru", "2017-12-25", "Merry Christmas beautiful!!!"));
        comments.add(new Comment("Naruto", "2018-01-01", "Happy New year ugly!!!"));
        comments.add(new Comment("Mulder", "2017-08-21", "Happy no day, whatever!!!"));
        comments.add(new Comment("Mulder", "2017-08-21", "This is a freaking long comment. or maybe not"));
        model.addAttribute("comments", comments);
        return "displayArticle";
    }
}
