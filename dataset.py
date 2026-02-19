"""
dataset.py — Labeled training corpus for fraud/threat detection.
Each entry: (text, label)  where label 0=legitimate, 1=fraud/threat
"""

DATASET = [
    # ══════════════════════════════════════════════════════════
    # LABEL 0 — LEGITIMATE
    # ══════════════════════════════════════════════════════════

    # Casual / personal
    ("Hey, are we still on for coffee at 3pm tomorrow?", 0),
    ("Thanks for dinner last night, it was lovely! Let's do it again soon.", 0),
    ("Can you pick up milk and bread on the way home?", 0),
    ("Happy birthday! Hope you have an amazing day filled with joy.", 0),
    ("Don't forget we have the family reunion on Sunday at noon.", 0),
    ("I just finished reading that book you recommended — absolutely brilliant.", 0),
    ("Running 10 minutes late, sorry! Start without me.", 0),
    ("The kids had a great time at the party. Thanks for organizing it.", 0),
    ("Thinking of you during this difficult time. Let me know if you need anything.", 0),
    ("Just checking in — how did the interview go yesterday?", 0),

    # Work / professional
    ("Reminder: the quarterly review is scheduled for Thursday at 2pm in conference room B.", 0),
    ("Please find attached the invoice for services rendered in October.", 0),
    ("Hi team, the standup meeting has been moved to 2pm today due to a conflict.", 0),
    ("Great work on the presentation today. The client was very impressed.", 0),
    ("The project deadline has been extended to the 20th. Please update your timelines.", 0),
    ("Could you review the attached document before our meeting tomorrow?", 0),
    ("Your leave request for December 24–26 has been approved.", 0),
    ("The quarterly report is ready for review in the shared drive.", 0),
    ("Please complete the mandatory cybersecurity training by end of month.", 0),
    ("I've uploaded the latest design mockups to the project folder.", 0),
    ("The budget proposal has been submitted to finance for review.", 0),
    ("We'll need your feedback on the draft contract by Friday at the latest.", 0),
    ("Our new office address will be effective from the 1st of next month.", 0),
    ("The all-hands meeting has been rescheduled to next Wednesday.", 0),

    # Transactional / notifications
    ("Your order #78432 has shipped and will arrive by Thursday.", 0),
    ("Your library books are due back on the 15th of this month.", 0),
    ("Your dentist appointment is confirmed for Friday at 10am with Dr. Smith.", 0),
    ("Your password was successfully changed. If this wasn't you, contact support.", 0),
    ("Your gym membership auto-renews on the 1st. No action required.", 0),
    ("Your flight BA2291 to New York departs at 08:15 from Terminal 5.", 0),
    ("Your subscription to Adobe Creative Cloud renews on November 1.", 0),
    ("Your tax documents for last year are ready to download from your account.", 0),
    ("We've updated our privacy policy. You can review the changes in your settings.", 0),
    ("Your parcel was delivered and left in a safe place as instructed.", 0),
    ("Your bank statement for September is now available in your online account.", 0),
    ("Reminder: your car MOT is due next month. Book online at our website.", 0),
    ("Your Amazon Prime membership was renewed for another year.", 0),
    ("Your Spotify subscription payment was processed successfully.", 0),
    ("We confirm your hotel reservation for 2 nights, check-in December 14.", 0),
    ("Your electricity bill for October is £87.43, due by the 20th.", 0),
    ("Your insurance renewal is due in 30 days. We'll send a reminder closer to the date.", 0),

    # Customer support (real)
    ("Thank you for contacting us. A support agent will respond within 24 hours.", 0),
    ("We're sorry to hear about your experience. Please reply with your order number.", 0),
    ("Your refund of £23.50 has been processed and will appear in 3–5 business days.", 0),
    ("We've resolved the issue you reported. Please let us know if it recurs.", 0),

    # News / informational
    ("The local council meeting will be held at the town hall on Monday at 7pm.", 0),
    ("Road closure on Main Street this weekend due to water main repairs.", 0),
    ("School will be closed on Friday for teacher training day.", 0),
    ("The community fundraiser raised over £5,000 for the local food bank.", 0),

    # Emails with links (legit)
    ("Please reset your password using this link: https://accounts.google.com/reset/password", 0),
    ("View your invoice online: https://stripe.com/invoices/in_1234567890", 0),
    ("Track your order here: https://www.amazon.co.uk/orders/track?id=123", 0),
    ("Join the meeting: https://zoom.us/j/123456789?pwd=abc123", 0),
    ("Download the report: https://docs.google.com/document/d/1a2b3c", 0),

    # ══════════════════════════════════════════════════════════
    # LABEL 1 — FRAUD / PHISHING / SCAM / THREAT
    # ══════════════════════════════════════════════════════════

    # Phishing — account takeover
    ("URGENT: Your bank account has been suspended! Verify immediately at http://secure-bank-login.xyz", 1),
    ("Your PayPal account is limited. Click here to restore access: http://paypal-secure.tk/verify", 1),
    ("Your Apple ID has been locked due to suspicious activity. Verify at: http://apple-id-verify.tk", 1),
    ("Dear customer, your Netflix subscription has expired. Update payment: http://netflix-billing.ml", 1),
    ("Your Amazon account will be closed! Confirm your details at http://amazon-verification.gq/login", 1),
    ("Microsoft account alert: Unusual sign-in detected. Secure your account: http://msft-secure.xyz", 1),
    ("Your Chase bank account has been temporarily frozen. Unlock at: http://chase-verify.top/login", 1),
    ("ALERT: Your Gmail password was changed. If not you, click here immediately: http://google-secure.tk", 1),
    ("Your Facebook account has been flagged. Restore access: http://fb-account-verify.xyz/restore", 1),
    ("IMPORTANT: Your Instagram was accessed from a new device. Verify: http://insta-secure.ml/check", 1),
    ("Your Coinbase account is at risk. Secure it NOW: http://coinbase-alert.tk/secure", 1),
    ("Wells Fargo: We detected suspicious activity. Please verify at http://wellsfargo-secure.gq", 1),
    ("Your Venmo account requires immediate verification: http://venmo-verify.xyz/confirm", 1),
    ("HSBC Security Alert: Please verify your identity at http://hsbc-online-secure.top/id-check", 1),
    ("Your Apple Pay has been disabled. Reactivate at http://applepay-secure.xyz/activate", 1),

    # Lottery / prize scams
    ("Congratulations! You've won $1,000,000 in our international lottery. Claim your prize now!", 1),
    ("You have been selected as our lucky winner! You've won a FREE iPhone 15! Claim: http://free-prize.xyz", 1),
    ("WINNER! You are our lucky customer #10000. Reply with your bank details to receive $5,000.", 1),
    ("Congratulations! Your email won the UK national lottery. Send your details to claim £850,000.", 1),
    ("You've won an exclusive reward! Claim your $500 gift card before it expires: http://reward-claim.tk", 1),
    ("FINAL DRAW: You are one of 5 winners of our $2 million sweepstakes! Respond immediately.", 1),
    ("Your survey reward of $750 is ready. Complete verification at http://survey-reward.xyz/claim", 1),
    ("Our records show you have unclaimed prize money of $4,500. Claim here: http://claim-funds.ml", 1),

    # IRS / government scams
    ("IRS FINAL NOTICE: You owe $3,200. Pay now or face ARREST. Call 1-800-555-IRS immediately.", 1),
    ("IRS ALERT: An arrest warrant has been issued. Pay $2,500 in gift cards to avoid jail.", 1),
    ("Your Social Security Number has been suspended. Call 1-888-555-9999 to reactivate immediately.", 1),
    ("SOCIAL SECURITY ADMINISTRATION: Your SSN has been involved in criminal activity. Call NOW.", 1),
    ("Department of Justice: You are under federal investigation. Contact agent immediately.", 1),
    ("IRS: Your tax refund of $4,200 is pending. Provide your bank routing number to receive funds.", 1),
    ("Medicare Alert: Your benefits will be cancelled unless you confirm your details today.", 1),
    ("Government grant of $9,800 available for you. No repayment! Apply: http://gov-grant.xyz", 1),

    # Nigerian prince / advance fee fraud
    ("DEAR FRIEND, I am a prince from Nigeria and need your help to transfer $45 million USD.", 1),
    ("I am the late General's widow. I have $28 million to transfer and need a trusted partner.", 1),
    ("Greetings, I am a diplomat with a consignment box worth $15.5 million needing your assistance.", 1),
    ("I have been diagnosed with cancer and wish to donate my $12 million estate to a trustworthy person.", 1),
    ("CONFIDENTIAL: I am an attorney for a deceased client who shares your surname. $6.5M is unclaimed.", 1),
    ("GOD BLESS YOU. I am contacting you regarding an inheritance of $4.2 million that needs a beneficiary.", 1),

    # Tech support scams
    ("Microsoft Security Alert: Your computer has been infected with 3 viruses! Call 1-800-555-MSFT NOW.", 1),
    ("WARNING: 5 critical errors found on your PC. Download our free antivirus: http://protect-now.xyz", 1),
    ("APPLE SECURITY: Your iPhone has been hacked. Call our specialist team: 1-888-APPLE-FIX", 1),
    ("Your computer is sending INFECTED data to the internet! Call support immediately: 1-800-555-1234", 1),
    ("Google Alert: Your Chrome browser has been compromised. Fix it now: http://chrome-fix.tk", 1),
    ("Norton Security: Your license has expired. Your PC is at risk. Renew at http://norton-renew.xyz", 1),

    # Financial / investment scams
    ("Crypto investment opportunity! Turn $500 into $50,000 in 30 days. 100% guaranteed returns!", 1),
    ("LIMITED TIME: Invest in our Bitcoin fund and earn 40% monthly returns. Minimum deposit $200.", 1),
    ("Work from home and earn $5,000 per week! No experience needed. Apply now for instant approval.", 1),
    ("Make $3,000 daily from home with our proven trading system. Free trial — sign up now!", 1),
    ("EXCLUSIVE: Early access to next 1000x crypto token. Buy now before it launches publicly!", 1),
    ("Forex trading signal: guaranteed 85% win rate. Join our VIP group for just $99/month.", 1),
    ("Your credit score can be ERASED legally! Remove all bad items for $299. Call now!", 1),
    ("Pre-approved for a $50,000 personal loan with NO credit check! Zero interest for 12 months!", 1),
    ("Debt relief program: eliminate 80% of your debt legally. Free consultation today!", 1),

    # Gift card / wire transfer scams
    ("Send $500 in iTunes gift cards and email me the codes. This is URGENT and confidential.", 1),
    ("I need you to purchase $1,000 in Google Play gift cards. I'll explain later. Very urgent.", 1),
    ("Please wire $2,000 to this account immediately. I'm stuck abroad and need help. — [Boss name]", 1),
    ("CEO REQUEST: Purchase $800 in Amazon gift cards for client gifts. Keep this confidential.", 1),
    ("Emergency: A family member has been arrested abroad. Send $3,000 bail via Western Union NOW.", 1),
    ("I am stranded in London after being robbed. Please wire £1,500 urgently. I'll repay you.", 1),

    # Delivery / parcel scams
    ("FINAL NOTICE: Unpaid toll of $2.34. Pay now to avoid $55 fine: http://pay-toll.xyz/now", 1),
    ("Your parcel is held at customs. Pay £2.99 clearance fee: http://royal-mail-customs.tk", 1),
    ("DHL: Your package could not be delivered. Pay £1.45 redelivery fee: http://dhl-redelivery.xyz", 1),
    ("FedEx Alert: Your shipment requires additional customs clearance. Pay at http://fedex-customs.ml", 1),
    ("USPS: Your parcel is on hold. Confirm your address and pay $3 fee: http://usps-track.xyz", 1),

    # Urgency / account suspension
    ("LAST CHANCE: Your account expires in 24 hours! Click immediately or lose all data: http://renew-now.gq", 1),
    ("URGENT: Your email account will be deleted in 12 hours unless you verify: http://email-verify.tk", 1),
    ("ALERT: Suspicious login from Russia detected on your account. Verify NOW or account deleted.", 1),
    ("Your account has been scheduled for permanent deletion. Cancel this action: http://cancel-delete.xyz", 1),
    ("WARNING: You have exceeded your storage limit. Your emails will be deleted. Upgrade: http://email-upgrade.ml", 1),
    ("Your subscription will auto-renew for $299 unless cancelled. Click: http://cancel-subscription.xyz", 1),

    # Romantic / relationship scams
    ("I saw your profile and felt an immediate connection. I am a US Army officer stationed abroad.", 1),
    ("Hello beautiful, I am a successful surgeon working with MSF. I would love to know you better.", 1),
    ("I know this is sudden but I have developed strong feelings for you. I need $3,000 for medical emergency.", 1),

    # COVID / health scams
    ("You qualify for a $1,400 COVID relief stimulus. Claim at http://covid-relief-fund.xyz now.", 1),
    ("FREE COVID-19 test kit — claim yours today. Provide your Medicare number: http://covid-test-free.tk", 1),
    ("VACCINE SIDE EFFECT COMPENSATION: You may be owed $5,000. File your claim: http://vax-claim.xyz", 1),

    # SMS smishing
    ("Your bank card has been used for a £450 transaction. If not you call 0800-555-FRAUD immediately.", 1),
    ("HMRC: You are owed a tax refund. Claim £289 at http://hmrc-refund.xyz before it expires.", 1),
    ("Your mobile account has been compromised. Verify at http://o2-secure.tk or service suspended.", 1),

    # Mixed urgency + financial
    ("FREE money from the government! You qualify for $8,500. Provide your SSN to claim today.", 1),
    ("URGENT: You have 48 hours to claim your inheritance of $780,000 before it is forfeited.", 1),
    ("Exclusive: You have been pre-selected for a $25,000 home improvement grant. Apply by midnight.", 1),
    ("Your student loan can be FORGIVEN today! Submit your details: http://loan-forgive.xyz/apply", 1),

    # ══════════════════════════════════════════════════════════
    # LABEL 0 — BORDERLINE LEGITIMATE (should score low)
    # ══════════════════════════════════════════════════════════
    ("Limited time offer: 30% off your next purchase. Use code SAVE30 at checkout.", 0),
    ("Don't miss our Black Friday sale — up to 50% off all products this weekend only.", 0),
    ("Your account login was attempted from a new device. Was this you? Click to confirm.", 0),
    ("Reminder: your annual subscription renews in 7 days. No action needed unless you wish to cancel.", 0),
    ("We noticed you haven't logged in recently. Click here to keep your account active.", 0),
    ("Your credit card ending in 4321 was charged $87.50 at Tesco. Not you? Contact us.", 0),
    ("Please verify your email address to complete your registration: https://signup.example.com/verify", 0),
    ("Your trial period ends in 3 days. Upgrade to keep access: https://app.example.com/upgrade", 0),
    ("You have a new message waiting in your inbox. Log in to read it.", 0),
    ("Join our referral program and earn $10 for every friend who signs up.", 0),

    # ══════════════════════════════════════════════════════════
    # LABEL 1 — BORDERLINE FRAUD (subtle)
    # ══════════════════════════════════════════════════════════
    ("We noticed unusual activity on your account. Please verify your recent transactions immediately.", 1),
    ("You may be eligible for a $500 refund. Confirm your personal details to receive it.", 1),
    ("Claim your exclusive membership reward of 50,000 points before they expire tonight.", 1),
    ("IMPORTANT: Your credit score has dropped 80 points. Fix it now: http://credit-fix.xyz", 1),
    ("Your debit card has been used at an ATM in another country. Confirm or block: http://card-secure.tk", 1),
    ("Congratulations! As a valued customer you've been selected for a special cash reward of $250.", 1),
    ("Act now — your loan application has been pre-approved. No credit check. Limited slots available.", 1),
    ("Verify your PayPal account to receive your pending payment of $1,247.50.", 1),
    ("Your identity has been used to open a new credit account. Dispute this immediately.", 1),
    ("WARNING: We found your email and password on the dark web. Secure your account now.", 1),
]

# Split into texts and labels
TEXTS  = [item[0] for item in DATASET]
LABELS = [item[1] for item in DATASET]

if __name__ == "__main__":
    fraud = sum(LABELS)
    legit = len(LABELS) - fraud
    print(f"Dataset: {len(DATASET)} samples  |  Fraud: {fraud}  |  Legit: {legit}")
