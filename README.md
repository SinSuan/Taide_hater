【抓含keyword文章】
format
https://ptt-search.nlpnchu.org/api/GetArticleByType?start={time}&end={time}&content={keyword}type=article&size={int}

ex
https://ptt-search.nlpnchu.org/api/GetArticleByType?start=1704443679&end=1705048481&content=%E6%9F%AF%E6%96%87%E5%93%B2&type=article&size=10000

time: unix time
keyword: BM25 關鍵字
int: 文章數


【抓文章的全部留言】
format
https://ptt-search.nlpnchu.org/api/GetCommentByArticle?article_id={article_id}

ex
https://ptt-search.nlpnchu.org/api/GetCommentByArticle?article_id=M.1704736938.A.965

article_id: 前一個抓回來內有一項是 article_id