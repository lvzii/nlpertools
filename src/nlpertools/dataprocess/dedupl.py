# 根据字段对一个元素为dict的list去重
def deduplicate_dict_list(dict_list: list, key: str) -> list:
    seen = set()
    result = []
    for d in dict_list:
        if key in d and d[key] not in seen:
            seen.add(d[key])
            result.append(d)
    return result
