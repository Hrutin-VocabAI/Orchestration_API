#!/usr/bin/env python3
"""
Hits each client's health-check endpoint (see Health_check_Service.py / the
/LIBAS_HEALTH route in app.py) and emails the team if anything fails.

Deliberately scheduled via cron rather than tied to orchestration_api.service's
own lifecycle - if the API itself is down, we still want this to fire and
report that as a failure, not go silent right when it matters most.

Run every 4 hours via cron (crontab -e):
    0 */4 * * * /usr/local/bin/python3 /media/vocab/DATA5/Vocab_Services_/DockerImage_CRED_ASR_/SURIYA/ORCHESTRATOR_API/Final_API/health_check_alert.py >> /media/vocab/DATA5/Vocab_Services_/DockerImage_CRED_ASR_/SURIYA/ORCHESTRATOR_API/Final_API/logs/health_check_alert.log 2>&1
"""
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import requests

HEALTH_ENDPOINTS = {
    "Libas": {
        # localhost only proves the process itself is alive - it can't catch a
        # firewall/NAT/network-path problem that would block a real external
        # client while the app is perfectly healthy. Checking the public IP
        # too (confirmed to actually route back to this box on this exact
        # port - see the curl test this was added from) verifies what a real
        # client experiences, not just that gunicorn is up.
        "localhost": "http://localhost:5000/LIBAS_HEALTH",
        "public_ip_27.111.72.61": "http://27.111.72.61:5000/LIBAS_HEALTH",
    },
}

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "suriya.kh@vocab-ai.com"
# Stored on the real ext4 filesystem (not the NTFS-mounted project drive) so
# permissions actually restrict access - see chmod 600 on this path.
SMTP_PASSWORD_FILE = Path("/home/vocab/.secrets/smtp_pass_orchestration_api.txt")
ALERT_FROM = SMTP_USER
ALERT_TO = ["suriya.kh@vocab-ai.com", "namrata.m@vocab-ai.com", "veeresh.h@vocab-ai.com"]

REQUEST_TIMEOUT = 180  # health check runs a real (short) transcription end-to-end, not an instant ping


def send_email(subject, body):
    password = SMTP_PASSWORD_FILE.read_text().strip()
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = ALERT_FROM
    msg["To"] = ", ".join(ALERT_TO)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, password)
        server.sendmail(ALERT_FROM, ALERT_TO, msg.as_string())


def check_route(route_label, url):
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        data = resp.json()
    except Exception as e:
        return "FAIL", [f"TRIGGER: UNREACHABLE - [{route_label}] {url} did not respond: {e}"]

    status = data.get("status", "FAIL")
    raw_triggers = data.get("triggers") or [f"TRIGGER: UNKNOWN - malformed health response: {data}"]
    triggers = [f"[{route_label}] {t}" for t in raw_triggers]
    return status, triggers


def main():
    for client_name, routes in HEALTH_ENDPOINTS.items():
        client_failed = False
        client_triggers = []

        for route_label, url in routes.items():
            status, triggers = check_route(route_label, url)
            print(f"[{client_name}][{route_label}] status={status}")
            for t in triggers:
                print(f"  {t}")
            if status != "PASS":
                client_failed = True
                client_triggers.extend(triggers)

        if client_failed:
            send_email(
                f"[ALERT] {client_name} health check FAILED",
                f"Health check for {client_name} failed on at least one route "
                f"(if only public_ip failed while localhost passed, suspect a "
                f"network/firewall path issue rather than the app itself):\n\n"
                + "\n".join(client_triggers),
            )


if __name__ == "__main__":
    main()
