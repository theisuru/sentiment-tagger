package com.isuru.bean;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class Comment {
	private int index;
	private String author;
	private String date;
	private String phrase;
	private Sentiment sentiment;
	
	public Comment() {
		
	}
	
	public Comment(String author, String date, String phrase) {
		this.author = author;
		this.date = date;
		this.phrase = phrase;
	}

	public Comment(int index, String author, String date, String phrase) {
		this.index = index;
		this.author = author;
		this.date = date;
		this.phrase = phrase;
		sentiment = Sentiment.UNDEFINED;
	}

	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
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

	public Sentiment getSentiment() {
		return sentiment;
	}

	public void setSentiment(Sentiment sentiment) {
		this.sentiment = sentiment;
	}

	@Override
	public String toString() {
		return "comment_" + index;
	}

}
