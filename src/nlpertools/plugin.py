#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
def sent_email(mail_user, mail_pass, receiver, title, content):
    """
    # mail_user = 'xxx'
    # mail_pass = 'xxx'
    # receiver = 'xxx'
    # sent_email(mail_user, mail_pass, receiver)
    """
    import smtplib
    from email.mime.text import MIMEText
    # log info
    mail_host = 'smtp.qq.com'
    mail_user = mail_user
    mail_pass = mail_pass
    sender = mail_user

    # email info
    message = MIMEText(content, 'plain', 'utf-8')
    message['Subject'] = title
    message['From'] = sender
    message['To'] = receiver

    # log and send
    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receiver, message.as_string())
        smtpObj.quit()
        print('send email succes')
    except smtplib.SMTPException as e:
        print('erro', e)
