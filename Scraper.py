# -*- coding: utf-8 -*-
"""
@author: mdorobek
"""
# Import
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas


# Getter methods for the attributes of the article
def get_html_content(url):
    request = requests.get(url)
    coverpage = request.content
    return BeautifulSoup(coverpage, 'html5lib')


def get_article(html_content):
    # the link of the article is written in the class "content title"
    return html_content.find_all(class_='content-title')


def get_link_list(coverpage_article):
    links = []
    for article in coverpage_article:
        links.append("http://www.musikreviews.de/" + article.a['href'])
    return links


def get_title(html_content):
    # in case of ERROR 404 no element will be returned
    try:
        return html_content.find('title').get_text().replace('"', "")
    except:
        return "ERROR: Cannot find a title"


def get_author(html_content):
    # in case of ERROR 404 no element will be returned
    try:
        return html_content.find('span', attrs={"itemprop": "reviewer"}).get_text().replace('"', "")
    except:
        return "ERROR: Cannot find a author"


def get_release_date(html_content):
    # in case of ERROR 404 no element will be returned
    try:
        return html_content.find('time', attrs={"itemprop": "dtreviewed"}).get_text().replace('"', "")
    except:
        # Return a invalid Date instead of a Error Message because it will be compared with the latest date in the csv
        return "31.12.9999"


def get_genre(html_content):
    # in case of ERROR 404 no element will be returned
    try:
        # The genre is written in a table
        rows = html_content.find(class_='review_info_table').findChildren("td")
        # write the genre in the line where the first collumn is "Stil:"
        for i in range(0, len(rows)):
            if rows[i].get_text() == "Stil:":
                position = i + 1
                # replace to eliminate all double quotes, they will be used as delimiter in the csv
                return rows[position].get_text().replace('"', "")
    except:
        return "ERROR: Cannot find a genre"


def get_article_text(html_content):
    list_paragraphs = []
    # in case of ERROR 404 no element will be returned
    try:
        content = html_content.find(class_='textcontent').find_all('p')
        # append all paragraphs
        for c in content:
            paragraph = c.get_text()
            list_paragraphs.append(paragraph)
    except:
        list_paragraphs.append("ERROR: Cannot find a text")
    final_article = " ".join(list_paragraphs)
    list_paragraphs.clear()  # clear the paragraph list to append the next article
    # eliminate all double quotes, they will be used as delimiter in the csv
    final_article = final_article.replace("\"", "\'")
    return final_article


def get_latest_date(csv):
    csv = pandas.read_csv(csv, encoding="ISO-8859-1")
    date_object_list = []
    for i in range(len(csv)):
        try:
            date_object_list.append(datetime.strptime(csv['release_date'][i], '%d.%m.%Y').date())
        except ValueError:
            pass
    return max(date_object_list)


def get_date_links(csv, date):
    date_link_list = []
    csv = pandas.read_csv(csv, encoding="ISO-8859-1")
    for i in range(len(csv)):
        try:
            if csv['release_date'][i] == date:
                date_link_list.append(csv['link'][i])
        except:
            pass
    return date_link_list


# Method to write the header of the csv-file
def write_header(filename):
    f = open(filename, "r")
    # if the header is not set in the first line it will be written
    if f.readline() != "title,link,author,genres,release_date,text\n":
        f.close()
        f = open(filename, "a")
        f.write("title,link,author,genres,release_date,text\n")
    f.close()


# Method to write the output csv
def write_csv(filename, title, link, author, genre, release_date, text):
    errors = 0
    write_header(filename)
    f = open(filename, "a")
    # if utf-8 encodeing doesn't work the counter of encode errors is increased
    try:
        if not (
                title == "ERROR: Cannot find a title" or author == "ERROR: Cannot find a author"
                or genre == "ERROR: Cannot find a genre" or release_date == "31.12.9999"
                or text == "ERROR: Cannot find a text"):
            f.write('"{}","{}","{}","{}","{}","{}"\n'.format(title, link, author, genre, release_date, text))
        else:
            errors += 1
    except:
        errors += 1
    f.close()
    return errors  # returns the number of encodeing errors which happended while the file was written


# Method to crawl through the webpages of the site musikreviews
def crawl_webpages(start_url, filename, logname):
    errors = 0
    all_len = 0
    start_time = datetime.now()
    url, page = start_url.split("=")
    page = int(page)
    go_on = True
    article_content_list = []
    release_date_list = []
    title_list = []
    latest_date = get_latest_date(filename)
    latest_page_link_list = []
    # crawl trough all pages while there is a next page or if the lastest date is reached
    while go_on:
        html_content = get_html_content(str(url) + "=" + str(page))
        go_on = html_content.find_all('a', title="Nächste News-Seite")
        link_list = get_link_list(get_article(html_content))
        for link in link_list:
            article_content_list.append(get_html_content(link))
        for art in article_content_list:
            release_date = get_release_date(art)
            # Set to false if the date of an article is earlier than the latest date of the articles in the csv
            go_on = datetime.strptime(release_date, '%d.%m.%Y').date() > latest_date
            release_date_list.append(release_date)
        if not go_on:
            for d in set(release_date_list):
                latest_page_link_list += get_date_links(filename, d)
        # print(latest_page_link_list)
        for link in range(len(link_list)):
            # print(link_list[l], " : ", link_list[l] not in latest_page_link_list)
            if link_list[link] not in latest_page_link_list:
                errors = write_csv(filename, get_title(article_content_list[link]), link_list[link],
                                   get_author(article_content_list[link]), get_genre(article_content_list[link]),
                                   release_date_list[link], get_article_text(article_content_list[link]))
                # increase the encodeing errors
                all_len += 1
        print("{} {}. Seite fertig. Aktuell {} Ergebnisse nach folgender Laufzeit {} und {} Fehler(n)".format(
            datetime.now(), page, all_len - errors, datetime.now() - start_time, errors))
        log = open(logname, "a")
        log.write("{} {}. Seite fertig. Aktuell {} Ergebnisse nach folgender Laufzeit {} und {} Fehler(n)\n".format(
            datetime.now(), page, all_len - errors, datetime.now() - start_time, errors))
        log.close()
        page += 1
        article_content_list.clear()
        release_date_list.clear()
        title_list.clear()


crawl_webpages("http://www.musikreviews.de/reviews/archiv/neue/?page=1", "musikreviews.csv", "log.log")
