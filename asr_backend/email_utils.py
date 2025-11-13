# asr_backend/email_utils.py
from __future__ import annotations
import os, smtplib, ssl
from email.message import EmailMessage

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")  # your account (e.g., Gmail)
SMTP_PASS = os.environ.get("SMTP_PASS")  # app password or smtp password
FROM_ADDR = os.environ.get("SMTP_FROM", SMTP_USER or "noreply@example.com")

def send_email(to: str, subject: str, body: str) -> None:
    """
    Sends a plain-text email. If SMTP_USER or SMTP_PASS is missing, prints a no-op
    line so dev/testing can continue without blowing up.
    """
    if not (SMTP_USER and SMTP_PASS):
        print(f"[email_utils] (noop) Would email {to!r}: {subject}\n{body}\n")
        return

    msg = EmailMessage()
    msg["From"] = FROM_ADDR
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

import os
import aiosmtplib
from email.message import EmailMessage

SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "false").lower() == "true"
SMTP_USERNAME = os.getenv("SMTP_USERNAME") or None
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or None
MAIL_FROM = os.getenv("MAIL_FROM", "Transcription App <no-reply@localhost>")

async def send_email(to: str, subject: str, html: str) -> None:
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(html, subtype="html")

    await aiosmtplib.send(
        msg,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        start_tls=SMTP_STARTTLS,
        username=SMTP_USERNAME,
        password=SMTP_PASSWORD,
        timeout=30,
    )
