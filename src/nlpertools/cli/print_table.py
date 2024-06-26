import os


def print_three_line_table(df):
    # TODO 这里需要添加可以支持excel里变红的功能
    import webbrowser

    # import pandas as pd
    # data = {'from_pc': ['valid_data', 'illegal_char', 'more_data'],
    #         'rom_pc': ['another_valid_data', 'illegal_char', 'data']}
    # df = pd.DataFrame(data)

    # 将 DataFrame 转换为 HTML 表格
    html_table = df.to_html(index=False)
    html_table = html_table.replace('border="1"', 'border="0"')

    first_line_px = str(2)
    second_line_px = str(1)
    third_line_px = str(2)
    # 定义三线表的 CSS 样式
    # // thead 表头
    # // tr 行
    # // td 单元格
    style = """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>页面标题</title>
    </head>
    <style>

        table {
            border-collapse: collapse;
        }

        tr, td, th {
            text-align: center; /* 水平居中文本 */
            vertical-align: middle; /* 垂直居中文本 */
        }
        thead tr {
            border-top: (first_line_px)px solid black;
            border-bottom: (second_line_px)px solid black;
        }

        thead th {
            border-bottom: (second_line_px)px solid black;
        }

        tbody tr td {
            border-bottom: 0px solid black;
        }

        tbody tr:last-child td {
            border-bottom: (third_line_px)px solid black;
        }
    </style>"""
    style = style.replace("(first_line_px)", first_line_px).replace("(second_line_px)", second_line_px).replace(
        "(third_line_px)", third_line_px)
    # 将 CSS 样式和 HTML 表格结合起来
    html = f"{style}{html_table}"
    print(html)
    temp_file_path = "temp.html"
    # 将 HTML 保存到文件中
    with open(temp_file_path, "w") as f:
        f.write(html)
    webbrowser.open('file://' + os.path.realpath(temp_file_path))


if __name__ == '__main__':
    print_three_line_table()
