import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from twilio.rest import Client
import pyttsx3
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import csv

# Function 1: Get Current Location
def get_current_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        location = data['loc'].split(',')
        lat, lng = location[0], location[1]
        print(f"Latitude: {lat}, Longitude: {lng}")
        print(f"Location: {data['city']}, {data['region']}, {data['country']}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function 2: Send Email
def send_email():
    sender_email = 'ryanreyo7906@gmail.com'
    password = 'cqoq vlpo cige hsdl'
    receiver_email = input("Enter the recipient's email address: ")
    subject = input("Enter the subject of the email: ")
    body = input("Enter the body of the email: ")
    smtp_server = "smtp.gmail.com"
    port = 587
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function 3: Scrape Google Search Results
def scrape_google(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    query = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={query}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.find_all('div', class_='g', limit=5):
        title_element = g.find('h3')
        link_element = g.find('a')
        if title_element and link_element:
            title = title_element.get_text()
            link = link_element['href']
            if link.startswith('/url?q='):
                link = link.split('/url?q=')[1].split('&')[0]
            results.append({"title": title, "link": link})
    return results

# Function 4: Send SMS using Twilio
def send_sms():
    account_sid = "AC58b22aedca9284cb6306111f7afeae5e"
    auth_token = "83a4c4f786193fcc599f4cb8c2ffa193"
    client = Client(account_sid, auth_token)
    from_phone = "+14326662708"
    to_phone = input("Enter the recipient's phone number (in E.164 format, e.g., +1234567890): ")
    message_body = input("Enter the SMS message: ")

    try:
        message = client.messages.create(
            body=message_body,
            from_=from_phone,
            to=to_phone
        )
        print(f"Message sent successfully with SID: {message.sid}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function 5: Text-to-Speech Conversion
def text_to_speech():
    engine = pyttsx3.init()
    text = input("Enter the text you want to convert to speech: ")
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

# Function 6: Set Volume
def set_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(volume_level, None)

# Function 7: Send Bulk Emails
def get_recipient_list(csv_file):
    recipients = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            email = row.get('email')
            if email:
                recipients.append(email)
    return recipients

def send_bulk_emails(sender_email, sender_password, recipient_list, subject, body):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    
    for recipient in recipient_list:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server.sendmail(sender_email, recipient, msg.as_string())
        print(f"Email sent to {recipient}")
    
    server.quit()

# Main Menu
def main_menu():
    while True:
        print("\nPlease choose an option:")
        print("1. Get Current Location")
        print("2. Send Email")
        print("3. Scrape Google Search Results")
        print("4. Send SMS using Twilio")
        print("5. Text-to-Speech Conversion")
        print("6. Set Volume")
        print("7. Send Bulk Emails")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            get_current_location()
        elif choice == '2':
            send_email()
        elif choice == '3':
            query = input("Enter your search query: ")
            top_results = scrape_google(query)
            if top_results:
                print("\nTop 5 Search Results:")
                for i, result in enumerate(top_results, 1):
                    print(f"{i}. {result['title']}\n   {result['link']}\n")
            else:
                print("No results found.")
        elif choice == '4':
            send_sms()
        elif choice == '5':
            text_to_speech()
        elif choice == '6':
            try:
                volume_level = float(input("Enter the volume level (0.0 to 1.0): "))
                if 0.0 <= volume_level <= 1.0:
                    set_volume(volume_level)
                    print(f"Volume set to {volume_level * 100}%")
                else:
                    print("Please enter a value between 0.0 and 1.0.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0.0 and 1.0.")
        elif choice == '7':
            csv_file = input("Enter the path to the recipients CSV file: ")
            subject = input("Enter the subject of the email: ")
            body = input("Enter the body of the email: ")
            recipient_list = get_recipient_list(csv_file)
            send_bulk_emails(sender_email, sender_password, recipient_list, subject, body)
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
