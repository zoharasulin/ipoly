import smtplib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Union
import imaplib
from plyer import notification
import os
from ipoly.traceback import raiser


def send(
    sender: str,
    password: str,
    receivers: Union[str, list[str]],
    subject: str,
    content: str,
    pngfiles: list[str] = (),
):
    """Send an email to (a/some) receiver(s).

    Args:
        sender: The sender email. Gmail and Outlook emails are supported.
        password: The sender's password.
        receivers: List or single email of the receiver(s).
        subject: The object of the email.
        content: The main text part of the email.
        pngfiles: List of attached files.

    """
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(receivers if receivers is list else [receivers])
    msg.attach(MIMEText(content, "plain"))
    for file in pngfiles:
        extension = file.split(r".")[-1]
        with open(file, "rb") as fp:
            if extension == "png":
                attached_file = MIMEImage(fp.read())
                # Open the files in binary mode.  Let the MIMEImage class automatically
                # guess the specific image type.
            else:
                attached_file = MIMEApplication(fp.read(), Name=file)
        msg.attach(attached_file)

    send_port = 587

    # Send the email via our own SMTP server.
    host = "smtp.gmail.com" if "gmail" in sender else "smtp.office365.com"
    with smtplib.SMTP(host, send_port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(sender, password)
        smtp.sendmail(sender, receivers, msg.as_string())
        smtp.quit()


def receive(receiver: str, password: str):
    # connect to the server and go to its inbox
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(receiver, password)
    # we choose the inbox but you can select others
    mail.select("inbox")
    # we'll search using the ALL criteria to retrieve
    # every message inside the inbox
    # it will return with its status and a list of ids
    status, data = mail.search(None, "ALL")
    # the list returned is a list of bytes separated
    # by white spaces on this format: [b'1 2 3', b'4 5 6']
    # so, to separate it first we create an empty list
    mail_ids = []
    # then we go through the list splitting its blocks
    # of bytes and appending to the mail_ids list
    for block in data:
        # the split function called without parameter
        # transforms the text or bytes into a list using
        # as separator the white spaces:
        # b'1 2 3'.split() => [b'1', b'2', b'3']
        mail_ids += block.split()

    # now for every id we'll fetch the email
    # to extract its content
    for i in mail_ids:
        # the fetch function fetch the email given its id
        # and format that you want the message to be
        status, data = mail.fetch(i, "(RFC822)")

        # the content data at the '(RFC822)' format comes on
        # a list with a tuple with header, content, and the closing
        # byte b')'
        for response_part in data:
            # so if its a tuple...
            if isinstance(response_part, tuple):
                # we go for the content at its second element
                # skipping the header at the first and the closing
                # at the third
                message = email.message_from_bytes(response_part[1])

                # with the content we can extract the info about
                # who sent the message and its subject
                mail_from = message["from"]
                mail_subject = message["subject"]

                # then for the text we have a little more work to do
                # because it can be in plain text or multipart
                # if its not plain text we need to separate the message
                # from its annexes to get the text
                if message.is_multipart():
                    mail_content = ""

                    # on multipart we have the text message and
                    # another things like annex, and html version
                    # of the message, in that case we loop through
                    # the email payload
                    for part in message.get_payload():
                        # if the content type is text/plain
                        # we extract it
                        if part.get_content_type() == "text/plain":
                            mail_content += part.get_payload()
                else:
                    # if the message isn't multipart, just extract it
                    mail_content = message.get_payload()

                # and then let's show its result
                print(f"From: {mail_from}")
                print(f"Subject: {mail_subject}")
                print(f"Content: {mail_content}")


def say(message: str) -> None:
    """Make the computer say the message out loud.

    This function works on Linux/Windows and Mac platforms.

    Args:
        message: The message the computer says.
            For security, the characters used must be in this list:\n
            0123456789,;:.?!-_ÂÃÄÀÁÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ

    """

    from string import ascii_letters

    if all(
        char
        in ascii_letters
        + "0123456789,;:.?!-_ÂÃÄÀÁÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        for char in message
    ):
        raiser(
            "This message will not be said because you used some symbols that may be used for doing some malicious injection :("
        )
    from platform import system as ps
    from os import system as oss

    match ps():
        case "Windows":
            from win32com.client import Dispatch

            speak = Dispatch("SAPI.SpVoice").Speak
            speak(message)
        case "Darwin":
            oss(f"say '{message}' &")
        case "Linux":
            oss(f"spd-say '{message}' &")
        case syst:
            raise RuntimeError("Operating System '%s' is not supported" % syst)


def notify(message: str, title: str = "Hey !") -> None:
    """Send a notification.

    This function works on Linux/Windows and Mac platforms and uses plyer in backend.

    Args:
        message: The message sent as a notification.
        title: The title of the notification. Defaults to 'Hey !'.

    """

    notification.notify(
        app_name="iPoly",
        app_icon=os.path.join(os.path.dirname(__file__), r"img\ipoly.ico"),
        title=title,
        message=message,
        ticker="iPoly ticker",
        timeout=15,
    )
