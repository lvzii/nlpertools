def estimate_cost(input_token_num, output_token_num, example_num=1, input_price=1, output_price=4):
    """
    估算成本
    :param input_token_num: 输入token数量
    :param output_token_num: 输出token数量
    :param example_num: 示例数量
    :param input_price: 输入token单价  / 1M
    :param output_price: 输出token单价 / 1M
    :return: 成本
    """
    price = (input_token_num * input_price + output_token_num * output_price) * example_num / 1000000
    print(f"Estimated cost: {price:.2f} 元")
    return price
