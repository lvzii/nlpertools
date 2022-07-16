#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import smtplib
from email.mime.text import MIMEText


class EmailClient(object):
    def __init__(self):
        self.mail_user = ""
        self.mail_pass = ""
        self.receiver = ""

    def sent_email(self, title, content):
        """
        # mail_user = 'xxx'
        # mail_pass = 'xxx'
        # receiver = 'xxx'
        # sent_email(mail_user, mail_pass, receiver)
        """

        # log info
        mail_host = 'smtp.qq.com'
        mail_user = self.mail_user
        mail_pass = self.mail_pass
        sender = mail_user

        # email info
        message = MIMEText(content, 'plain', 'utf-8')
        message['Subject'] = title
        message['From'] = sender
        message['To'] = self.receiver

        # log and send
        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect(mail_host, 25)
            smtpObj.login(mail_user, mail_pass)
            smtpObj.sendmail(sender, self.receiver, message.as_string())
            smtpObj.quit()
            print('send email succes')
        except smtplib.SMTPException as e:
            print('erro', e)
