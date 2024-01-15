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

def get_address_2_article(time_start, time_end, keyword, num_article):
    home_addr_article = "https://ptt-search.nlpnchu.org/api/GetArticleByType"
    time_component = f"?start={int(time_start)}&end={int(time_end)}"
    info_component = f"&content={keyword}&type=article&size={num_article}"
    return home_addr_article + time_component + info_component

def get_address_2_comment(article_id):
    home_addr_comment = "https://ptt-search.nlpnchu.org/api/GetCommentByArticle"
    article_component = f"?article_id={article_id}"
    return home_addr_comment + article_component

def get_total_article(address_2_article):
    r = requests.get(address_2_article)
    r = r.json()

    total_article = {}
    for hit in r["hits"]:
      hit_source = hit["_source"]

      current_article = {}
      current_article["article_title"] = hit_source["article_title"]
      current_article["content"] = hit_source["content"]
      
      article_id = hit_source["article_id"]
      total_article[article_id] = current_article

    return total_article

def get_total_comment(total_article):
    print("\n\nenter get_total_comment\n\n")
    
    total_article_id = total_article.keys()
    print(f"total_article_id = {total_article_id}")
    
    total_comment = {}
    for article_id in total_article_id:
        address_2_comment = get_address_2_comment(article_id)
        print(f"{article_id}\taddress_2_comment = {address_2_comment}")
        
        r = requests.get(address_2_comment)
        r = r.json()

        total_comment_4_article_id = []
        for hit in r["hits"]:
            hit_source = hit["_source"]

            current_comment_4_article_id = {}
            current_comment_4_article_id["article_title"] = hit_source["article_title"]
            current_comment_4_article_id["content"] = hit_source["content"]
            
            total_comment_4_article_id.append(current_comment_4_article_id)
            
        total_comment[article_id] = total_comment_4_article_id
    
    print("\n\nexit get_total_comment\n\n")
    return total_comment

def concat_prompt(article, comment):
    """
    formal    :  f"<s> [INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    train     :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    inference :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST]"
    """
    prompt_sys = "你是一個酸民，看完文章："
    prompt_full = f"[INST] <SYS>{prompt_sys}</SYS>「{article}」會回答：[/INST] {comment} </s>"
    return prompt_full

# def create_dataset_4_article_id(article_4_article_id, total_comment_4_article_id):
#     dataset_for_article_id = []
#     for comment_4_article_id in total_comment_4_article_id:
#         data = concat_prompt(article_4_article_id, comment_4_article_id)
#         dataset_for_article_id.append(data)
#     return dataset_for_article_id

def create_dataset(total_article, total_comment):
    total_article_id = total_article.keys()
    _dataset_ = []
    for article_id in total_article_id:
        article = total_article[article_id]
        total_comment_4_article = total_comment[article_id]
        for comment_4_article in total_comment_4_article:
            data = concat_prompt(article["content"], comment_4_article["content"])
            _dataset_.append(data)
    return _dataset_

def main():
    UNIX_sta = datetime.datetime(YEAR, MONTH, DAY, HOUR, MINUTE-1, SECOND)
    UNIX_end = datetime.datetime(YEAR, MONTH, DAY, HOUR, MINUTE+1, SECOND)

    UNIX_sta_stamp = (UNIX_sta - datetime.timedelta(hours=8)).timestamp()
    UNIX_end_stamp = (UNIX_end - datetime.timedelta(hours=8)).timestamp()
    
    address_2_article = get_address_2_article(UNIX_sta_stamp, UNIX_end_stamp, KEYWORD, 5)
    print(f"address_2_article = {address_2_article}")
    
    total_article = get_total_article(address_2_article)
    print(f"len(total_article) = {len(total_article)}")
    total_article_id = list(total_article.keys())
    print(f"""total_article[total_article_id[0]] = {total_article[total_article_id[0]]}""")
    
    total_comment = get_total_comment(total_article)
    print(f"""len(total_article) = {len(total_comment)}""")
    print(f"""len(total_article[total_article_id]) =
          {len(total_comment[total_article_id[0]])}""")
    print(f"""len(total_article[total_article_id]) =
          {total_comment[total_article_id[0]][0]}""")
    
    _dataset_ = create_dataset(total_article, total_comment)
    print(f"_dataset_ = {_dataset_}")
    
    with open("example/test.json", "w", encoding="utf-8") as json_file:
        json.dump(_dataset_, json_file, ensure_ascii=False)
    


if __name__ == "__main__":
    main()