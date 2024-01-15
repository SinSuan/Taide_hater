import requests
from bs4 import BeautifulSoup
import json
import datetime

YEAR = 2024
MONTH = 1
DAY = 12
HOUR = 11
MINUTE = 47
SECOND = 10

KEYWORD = "賴清德"

def address_2_article(time_start, time_end, keyword, num_article):
    home_addr_article = "https://ptt-search.nlpnchu.org/api/GetArticleByType"
    time_component = f"?start={int(time_start)}&end={int(time_end)}"
    info_component = f"&content={keyword}&type=article&size={num_article}"
    return home_addr_article + time_component + info_component

def address_2_comment(article_id):
    home_addr_comment = "https://ptt-search.nlpnchu.org/api/GetCommentByArticle"
    article_component = f"?article_id={article_id}"
    return home_addr_comment + article_component

def get_total_article_or_comment(address):
    r = requests.get(address)
    r = r.json()

    total_article_or_comment = []
    for hit in r["hits"]:
      hit_source = hit["_source"]

      current_article_or_comment = {}
      current_article_or_comment["article_id"] = hit_source["article_id"]
      current_article_or_comment["article_title"] = hit_source["article_title"]
      current_article_or_comment["content"] = hit_source["content"]
      
      total_article_or_comment.append(current_article_or_comment)

    return total_article_or_comment

def concat_prompt(article, comment):
    """
    formal    :  f"<s> [INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    train     :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    inference :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST]"
    """
    prompt_sys = "你是一個酸民，看完文章："
    prompt_full = f"[INST] <SYS>{prompt_sys}</SYS>「{article}」會回答：[/INST] {comment} </s>"
    return prompt_full

def create_dataset_for_article_id(article_id):
    dataset_for_article_id = []
    for
    
    return dataset_for_article_id

def main():
    UNIX_sta = datetime.datetime(YEAR, MONTH, DAY, HOUR, MINUTE-1, SECOND)
    UNIX_end = datetime.datetime(YEAR, MONTH, DAY, HOUR, MINUTE+1, SECOND)

    UNIX_sta_stamp = (UNIX_sta - datetime.timedelta(hours=8)).timestamp()
    UNIX_end_stamp = (UNIX_end - datetime.timedelta(hours=8)).timestamp()
    
    print(address_2_article(UNIX_sta_stamp, UNIX_end_stamp, KEYWORD, 1))
    
    


if __name__ == "__main__":
    main()