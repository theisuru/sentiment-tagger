package com.isuru.bean;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;
import java.util.ArrayList;
import java.util.List;

@XmlRootElement
@XmlType(propOrder = {"articleId", "title", "author", "date", "body", "comments", "url"})
public class NewsArticle {
    private long articleId;
    private String author;
    private String date;
    private String title;
    private String body;
    private List<Comment> comments;
    private String url;

    public NewsArticle() {
        comments = new ArrayList<>();
    }

    public NewsArticle(String title, String body, List<Comment> comments) {
        this.title = title;
        this.body = body;
        this.comments = comments;
    }

    public NewsArticle(long articleId, String title, String body, List<Comment> comments) {
        this.articleId = articleId;
        this.title = title;
        this.body = body;
        this.comments = comments;
    }

    public long getArticleId() {
        return articleId;
    }

    public void setArticleId(long articleId) {
        this.articleId = articleId;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }


    @XmlElementWrapper(name = "comments")
    @XmlElement(name = "comment")
    public List<Comment> getComments() {
        return comments;
    }

    public void setComments(List<Comment> comments) {
        this.comments = comments;
    }

    public void addComment(Comment comment) {
        comments.add(comment);
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String toTxt() {
        StringBuilder sb = new StringBuilder();
        sb.append(title).append("\n")
                .append(body).append("\n");
        for (Comment c : comments) {
            sb.append("\n\n").append(c.toTxt());
        }
        return sb.toString();
    }

    public String toTxtComments() {
        StringBuilder sb = new StringBuilder();
        sb.append(title).append("\n");
        for (Comment c : comments) {
            sb.append("\n\n").append(c.toTxt());
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(author).append("\n").append(date).append("\n").append(title).append("\n");
        if (body != null && body.length() >= 100)
            sb.append(body.substring(0, 100)).append(".....");
        else
            sb.append(body);
        sb.append("\n# commets: ").append(comments.size());
        return sb.toString();
    }

    /*@Override
    public boolean equals(Object o) {
         return o instanceof NewsArticle && articleId == ((NewsArticle) o).getArticleId();
    }

    @Override
    public int hashCode() {
        return (int) articleId;
    }*/


}
