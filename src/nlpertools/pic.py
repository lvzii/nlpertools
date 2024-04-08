#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from io import BytesIO


def convert_pic_dpi(path):
    from PIL import Image

    img = Image.open(path)
    w, h = img.size
    rate = 0.1
    img = img.resize((int(w * rate), int(h * rate)))
    img.save("test.jpg")  # （224，224）


def image2binary(image):
    """
    image: PIL.image
    """
    # 假设你已经有了一个Image对象
    # image = Image.open('a.png')
    # 创建一个BytesIO对象来保存二进制数据
    buffered = BytesIO()
    # 保存Image对象到BytesIO对象，确保使用正确的格式
    image.save(buffered, format="JPEG")
    # 获取二进制数据
    binary_data = buffered.getvalue()
    # 确保输出缓冲区被重置，以便后续使用
    buffered.seek(0)
    # 现在，binary_data包含了完整的JPEG图像数据
    # 你可以将这个数据发送到网络请求，或者保存到文件
    # with open('aa.jpg', 'wb') as f:
    #     f.write(binary_data)
    return binary_data


def invert_colors(image_path, output_path):
    from PIL import Image, ImageOps

    image = Image.open(image_path)
    black_and_white = image.convert("L")

    # 对调黑白颜色
    inverted = ImageOps.invert(black_and_white)

    # 保存修改后的图片
    inverted.save(output_path)


def pdf2pic(path):
    from pdf2image import convert_from_path

    pages = convert_from_path(path, 500)

    # 保存
    num = 1
    for page in pages:
        page.save("out{}.jpg".format(num), "JPEG")
        num += 1


def concat_image():
    import numpy as np

    from PIL import Image

    # 这里是需要合并的图片路径
    paths = ["out{}.jpg".format(i) for i in range(1, 14)]
    img_array = ""
    img = ""
    for i, v in enumerate(paths):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)

    # 保存图片
    img.save("图1.jpg")


class DrawDesktopBackground:
    @staticmethod
    def generate_image(text1, text2, text3, text4, color1, color2, color3, color4):
        # 不支持中文
        # 样式参考 https://zhuanlan.zhihu.com/p/365624498
        from PIL import Image, ImageDraw, ImageFont

        # 配色方案1：
        # '#1a4b61', '#f47678', '#79a863', '#a8a8a8'
        # Create image object with black background
        # 创建黑色背景的图像对象
        img = Image.new("RGB", (1920, 1080), color="black")

        # Create draw object
        # 创建绘图对象
        draw = ImageDraw.Draw(img)

        # Define font and font size
        # 定义字体和字体大小
        font = ImageFont.truetype("arial.ttf", size=100)

        # Define text color
        # 定义文本颜色
        text_color = (255, 255, 255)

        # Define rectangle coordinates
        # 定义矩形坐标
        rect1 = (0, 0, 960, 540)
        rect2 = (960, 0, 1920, 540)
        rect3 = (0, 540, 960, 1080)
        rect4 = (960, 540, 1920, 1080)

        # Draw rectangles
        # 绘制矩形
        draw.rectangle(rect1, fill=color1)
        draw.rectangle(rect2, fill=color2)
        draw.rectangle(rect3, fill=color3)
        draw.rectangle(rect4, fill=color4)
        # 通过边框和裁剪实现样式3
        # draw.rectangle(rect4, fill=color4, outline="white", width=2)
        # img = img.crop(box=(2, 2, 1919, 1079))
        # Draw text in rectangles
        # 在矩形中绘制文本
        draw.text((480, 270), text1, font=font, fill=text_color, anchor="mm")
        draw.text((1440, 270), text2, font=font, fill=text_color, anchor="mm")
        draw.text((480, 810), text3, font=font, fill=text_color, anchor="mm")
        draw.text((1440, 810), text4, font=font, fill=text_color, anchor="mm")

        # Save image
        # 保存图像
        img.save("generated_image.png")

        # generate_image('Text 1', 'Text 2', 'Text 3', 'Text 4', '#F0E68C', '#ADD8E6', '#98FB98', '#FFC0CB')

    @staticmethod
    def generate_image_style_2(
        text1, text2, text3, text4, color1, color2, color3, color4
    ):
        # 不支持中文
        # 样式参考 https://zhuanlan.zhihu.com/p/365624498
        from PIL import Image, ImageDraw, ImageFont

        # Create image object with white background
        bg_width, bg_height = 1920, 1080
        rate = 0.5
        # 小矩形的宽何高
        width, height = 400, 100
        # 小矩形距离底边的距离
        margin = height
        # 文本距离小框左侧的距离
        text_left_margin = 50
        # 文本距离小框的上边距
        text_up_margin = 12
        font_size = 65
        width, height, margin, text_up_margin, text_left_margin, font_size = (
            width * rate,
            height * rate,
            margin * rate,
            text_up_margin * rate,
            text_left_margin * rate,
            int(font_size * rate),
        )
        # Define font
        font = ImageFont.truetype("arial.ttf", size=font_size)

        # Create drawing object
        img = Image.new("RGB", (bg_width, bg_height), color="white")
        draw = ImageDraw.Draw(img)

        # Draw rectangles
        big_rect1 = (0, 0, 960, 540)
        big_rect2 = (960, 0, 1920, 540)
        big_rect3 = (0, 540, 960, 1080)
        big_rect4 = (960, 540, 1920, 1080)
        draw.rectangle(big_rect1, fill=color1)
        draw.rectangle(big_rect2, fill=color2)
        draw.rectangle(big_rect3, fill=color3)
        draw.rectangle(big_rect4, fill=color4)

        # Draw small rectangles in corners
        small_rect1 = (0, margin, width, height + margin)
        small_rect2 = (bg_width - width, margin, bg_width, height + margin)
        small_rect3 = (0, bg_height - margin - height, width, bg_height - margin)
        small_rect4 = (
            bg_width - width,
            bg_height - margin - height,
            bg_width,
            bg_height - margin,
        )
        draw.rectangle(small_rect1, fill="white")
        draw.rectangle(small_rect2, fill="white")
        draw.rectangle(small_rect3, fill="white")
        draw.rectangle(small_rect4, fill="white")

        # Draw text in rectangles
        text_point1 = (text_left_margin, margin + text_up_margin)
        text_point2 = (text_left_margin + bg_width - width, margin + text_up_margin)
        text_point3 = (text_left_margin, bg_height - margin - height + text_up_margin)
        text_point4 = (
            text_left_margin + bg_width - width,
            bg_height - margin - height + text_up_margin,
        )
        draw.text(text_point1, text1, font=font, fill=color1)
        draw.text(text_point2, text2, font=font, fill=color2)
        draw.text(text_point3, text3, font=font, fill=color3)
        draw.text(text_point4, text4, font=font, fill=color4)

        # Save image
        img.save("generated_image.png")

        # generate_image('OpenSource', 'Doing', 'Fixed', 'Tmp',
        #            "#1a4b61", "#a8a8a8", "#f47678", "#fad048")

    @staticmethod
    def generate_image_style_3(
        text1, text2, text3, bg_color, rec_color, text_color, pic
    ):
        # 样式参考小红书 http://xhslink.com/f1JBTp
        from PIL import Image, ImageDraw, ImageFont

        # Create image object with white background
        bg_width, bg_height = 1920, 1080

        rate = 0.5
        font_size = 50
        text_rec_distance = 20
        font_size = int(font_size * rate)
        # Define font
        font = ImageFont.truetype("arial.ttf", size=font_size)

        # Create drawing object
        img = Image.new("RGB", (bg_width, bg_height), color=bg_color)
        draw = ImageDraw.Draw(img)

        margin_up = 60
        margin_left = margin_right = 50
        rec_im_distance = -50
        rec_rec_distance = 45
        rec2_width = 600
        rec2_height = 500
        rec1_width, rec1_height = 600, bg_height - margin_up * 2

        rec1_x, rec1_y = margin_left, margin_up
        rec2_x, rec2_y = rec1_x + rec1_width + rec_rec_distance, rec1_y
        rec3_x, rec3_y, rec3_width, rec3_height = (
            rec2_x,
            rec2_y + rec2_height + rec_rec_distance,
            -1,
            -1,
        )
        im_width = im_height = 600

        # Insert Pic
        im = Image.open(pic)
        im = im.resize((im_width, im_height))
        img.paste(im, (rec2_x + rec2_width + rec_im_distance, margin_up))
        # Draw rectangles
        big_rect1 = (rec1_x, rec1_y, rec1_x + rec1_width, rec1_y + rec1_height)
        big_rect2 = (rec2_x, rec2_y, rec2_x + rec2_width, rec2_y + rec2_height)
        big_rect3 = (rec3_x, rec3_y, bg_width - margin_right, bg_height - margin_up)
        draw.rectangle(big_rect1, fill=rec_color)
        draw.rectangle(big_rect2, fill=rec_color)
        draw.rectangle(big_rect3, fill=rec_color)

        # Draw text in rectangles
        text_point1 = (rec1_x + text_rec_distance, rec1_y + text_rec_distance)
        text_point2 = (rec2_x + text_rec_distance, rec2_y + text_rec_distance)
        text_point3 = (rec3_x + text_rec_distance, rec3_y + text_rec_distance)

        draw.text(text_point1, text1, font=font, fill=text_color)
        draw.text(text_point2, text2, font=font, fill=text_color)
        draw.text(text_point3, text3, font=font, fill=text_color)

        # Save image
        img.save("generated_image.png")

        # generate_image_style_3('· OpenSource ·', '· Doing ·', '· Fixed ·',
        #                        "#e8e8e8", "#dfdfdf", "#707070", "cat.jpg")

    @staticmethod
    def generate_from_pic():
        # 通过版面识别识别出框所在的位置，
        pass
