def send_email(receiver_email, otp):
    # Dev mode: just print OTP to console and log it
    print(f"[OTP] To: {receiver_email} | OTP: {otp}")
    with open("otps.log", "a") as f:
        f.write(f"{receiver_email}\t{otp}\n")
    return True
