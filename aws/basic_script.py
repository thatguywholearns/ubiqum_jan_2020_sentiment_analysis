#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import urllib.request
import re
from textblob import TextBlob

url = "https://data.ubiqum.com/"

page = urllib.request.urlopen(url)

soup = BeautifulSoup(page, 'html.parser')

content_lis = soup.find_all('div', attrs={'class': 'page-inner'})

for x in content_lis:
    text = x.find('p').text

print("\n", text, "\n")


blob = TextBlob(text)

print(blob.sentiment.polarity, "\n")


def split_line(text):
	words = text.split()
	for word in words:
		s = TextBlob(word)
		print(word, s.sentiment.polarity)
	
split_line(text)
