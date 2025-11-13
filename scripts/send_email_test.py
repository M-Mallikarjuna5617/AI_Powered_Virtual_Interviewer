"""Self-contained test: start a local aiosmtpd debug SMTP server, send a test OTP
and print the server-received message plus the send result.

This script will try to install `aiosmtpd` into the current Python environment
if it's not already available. It starts a temporary SMTP server on localhost:1025.

Usage:
  python scripts/send_email_test.py target@example.com

Notes:
  - This does not deliver to real mailboxes. It prints the message received by
    the local debug server so you can verify the content.
"""

import sys
import time
import subprocess

from utils import send_email


def ensure_aiosmtpd():
    try:
        import aiosmtpd  # noqa: F401
        return True
    except Exception:
        print("aiosmtpd not found; attempting to install it into the active environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "aiosmtpd"])
            print("aiosmtpd installed.")
            return True
        except Exception as e:
            print("Failed to install aiosmtpd:", e)
            return False


def run_test(target_email: str):
    # ensure aiosmtpd is present
    if not ensure_aiosmtpd():
        print("Please install aiosmtpd manually (pip install aiosmtpd) and re-run the script.")
        return

    from aiosmtpd.controller import Controller

    class PrintingHandler:
        async def handle_DATA(self, server, session, envelope):
            print("\n----- Debug SMTP received message -----")
            print("From:", envelope.mail_from)
            print("To:", envelope.rcpt_tos)
            try:
                content = envelope.content.decode('utf8', errors='replace')
            except Exception:
                content = str(envelope.content)
            print(content)
            print("----- End message -----\n")
            return '250 Message accepted for delivery'

    controller = Controller(PrintingHandler(), hostname='127.0.0.1', port=1025)
    try:
        controller.start()
    except Exception as e:
        print('Failed to start local SMTP server:', e)
        print('If another process is using port 1025, either stop it or set MAIL_PORT to a free port in .env')
        return

    try:
        print('Local debug SMTP server started on 127.0.0.1:1025')
        # small pause for server to be fully ready
        time.sleep(0.2)
        result = send_email(target_email, '000000')
        print('send_email returned', result)
        if not result:
            print('send_email reported failure. Check .env and MAIL_* environment variables (or app logs).')

        # give the controller a moment to print received message
        time.sleep(0.5)

    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/send_email_test.py target@example.com")
        sys.exit(1)
    target = sys.argv[1]
    run_test(target)
