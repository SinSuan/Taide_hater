"""
requests articles and comments from ptt
save to DIR_4_DATASET
"""

import datetime
import json
import requests

YEAR = 2024
MONTH = 1
DAY = 12
HOUR = 11
MINUTE = 47
SECOND = 10

KEYWORD = "賴清德"

DIR_4_DATASET = "data/"

def get_unix_time_stamp(year, month, day, hour, minute, second):
    """
    (None)
    """
    tw_time = datetime.datetime(year, month, day, hour, minute, second)

    # '- datetime.timedelta(hours=8)' for getting the time zone of england
    unix_time = tw_time - datetime.timedelta(hours=8)

    unix_time_stamp = (unix_time - datetime.timedelta(hours=8)).timestamp()
    unix_time_stamp = int(unix_time_stamp)

    return unix_time_stamp

def get_search_range()-> [int, int]:
    """
    use the global var:
        YEAR, MONTH, DAY, HOUR, MINUTE, SECOND
    """

    unix_sta_stamp = get_unix_time_stamp(YEAR, MONTH, DAY, HOUR, MINUTE-1, SECOND)
    unix_end_stamp = get_unix_time_stamp(YEAR, MONTH, DAY, HOUR, MINUTE+1, SECOND)

    return unix_sta_stamp, unix_end_stamp


def get_address_2_article(time_start, time_end, keyword, num_article):
    """
    keyword is for BM25
    """
    home_addr_article = "https://ptt-search.nlpnchu.org/api/GetArticleByType"
    time_component = f"?start={int(time_start)}&end={int(time_end)}"
    info_component = f"&content={keyword}&type=article&size={num_article}"
    return home_addr_article + time_component + info_component

def get_address_2_comment(article_id):
    """
    (None)
    """
    home_addr_comment = "https://ptt-search.nlpnchu.org/api/GetCommentByArticle"
    article_component = f"?article_id={article_id}"
    return home_addr_comment + article_component

def get_total_article(address_2_article):
    """
    get articles about 1 topic
    
    Return
    ------
    {
        article_id_1: "str1",
        article_id_2: "str2",
        ...
    }
    """
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
    """
    get comments about 1 article
    
    Return
    ------
    {
        article_id_1: ["str11", "str12", "str13", ...],
        article_id_2: ["str21", "str22", "str23", ...],
        ...
    }
    """
    print("\n\n\t***enter get_total_comment***\n\n")

    total_article_id = total_article.keys()
    print(f"""total_article_id =
          {total_article_id}""")

    total_comment = {}
    for article_id in total_article_id:
        address_2_comment = get_address_2_comment(article_id)
        print(f"""{article_id}\taddress_2_comment =
              {address_2_comment}""")

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

    print("\n\n\t***exit get_total_comment***\n\n")
    return total_comment

def concat_prompt(article, comment):
    """
        *重要：<s>、[INST]、<SYS>、</s>、[/INST]、</SYS> 前後都要空格，不然模型抓不到這些 token*
    formal    :  f"<s> [INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    train     :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST] {answer} </s>"
    inference :      f"[INST] <SYS> {system_promt} </SYS> {user_promt} [/INST]"
    """
    prompt_sys = "你是一個酸民，看完文章："
    prompt_full = f" [INST] <<SYS>> {prompt_sys} <</SYS>> 「{article}」會回答： [/INST] {comment} </s>"
    return prompt_full

def create_dataset(total_article, total_comment):
    """
    Return
    ------
    [
        {
            "text": "str1"
        },
        {
            "text": "str2"
        },
        ...
    ]
    """
    total_article_id = total_article.keys()
    _dataset_ = []
    for article_id in total_article_id:
        article = total_article[article_id]
        total_comment_4_article = total_comment[article_id]
        for comment_4_article in total_comment_4_article:
            data = concat_prompt(article["content"], comment_4_article["content"])
            _dataset_.append({"text":data})
    return _dataset_

def main():
    """main function
    create and save a dataset to DIR_4_DATASET
    """
    unix_sta_stamp, unix_end_stamp = get_search_range()

    address_2_article = get_address_2_article(unix_sta_stamp, unix_end_stamp, KEYWORD, 5)
    # print(f"""address_2_article =
    #       {address_2_article}""")

    total_article = get_total_article(address_2_article)
    # print(f"""len(total_article) =
    #       {len(total_article)}""")
    # total_article_id = list(total_article.keys())
    # print(f"""total_article[total_article_id[0]] =
    #       {total_article[total_article_id[0]]}""")

    total_comment = get_total_comment(total_article)
    # print(f"""len(total_article) =
    #       {len(total_comment)}""")
    # print(f"""len(total_article[total_article_id]) =
    #       {len(total_comment[total_article_id[0]])}""")
    # print(f"""len(total_article[total_article_id]) =
    #       {total_comment[total_article_id[0]][0]}""")

    _dataset_ = create_dataset(total_article, total_comment)
    # print(f"_dataset_[0] = {_dataset_[0]}")

    with open(DIR_4_DATASET + "test.json", "w", encoding="utf-8") as json_file:
        json.dump(_dataset_, json_file, ensure_ascii=False)

if __name__ == "__main__":
    main()
    