package com.isuru.crawler.refactor;

import com.isuru.bean.Comment;
import com.isuru.bean.Sentiment;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name = "comment")
public class OldComment2 {
    private String author;
    private String date;
    private String phrase;

    public OldComment2() {
        //default constructor
    }

    public OldComment2(String author, String date, String phrase) {
        this.author = author;
        this.date = date;
        this.phrase = phrase;
    }

    public Comment getNewerVersion(int index) {
        phrase = phrase.trim().replaceAll("\\(.{1,5}\\)$", "").trim();
        Comment comment = new Comment(index, author, date, phrase);
        comment.setSentiment(Sentiment.UNDEFINED);
        return comment;
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

    public String getPhrase() {
        return phrase;
    }

    public void setPhrase(String phrase) {
        this.phrase = phrase;
    }

    public String toTxt() {
        return phrase;
    }

    @Override
    public String toString() {
        return "Author: " + author + ", Date: " + date + "Pharse: " + phrase;
    }

    public static void main(String[] args) {
        String test = "මම කියන්නේ හුසේන් මරා දාපු (නිවි) එක නම් හොඳයි. නමුත් එය කරපු විදිහයි වැරදියි. (නිි)";
        String test1 = test.replaceAll("\\(.{1,5}\\)$", "");

        System.out.println(test);
        System.out.println(test1);
    }

}
