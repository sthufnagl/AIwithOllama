import ollama
from ollama import generate
from geopy.geocoders import Nominatim
import glob
import pandas as pd
from PIL import Image
from PIL import ExifTags
import os
from io import BytesIO


def get_location(latitude, longitude):
    """Gibt den Ort und das Land für gegebene Koordinaten zurück."""
    if latitude is None or longitude is None:
        return None, None  # Keine Koordinaten verfügbar

    geolocator = Nominatim(user_agent="ragImages")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)

    if location:
        address = location.raw.get('address', {})
        city = address.get('city', '') or address.get('town', '') or address.get('village', '')
        country = address.get('country', '')
        print("####################################################")
        print(f"Ort: {city}, Land: {country}")
        print("####################################################")
        return city, country
    else:
        return None, None


def get_exif_data(img):
    """Extrahiere EXIF-Daten aus einem PIL Image-Objekt."""
    try:
        exif_data = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
    except AttributeError:
        exif_data = {}
    return exif_data


def load_or_create_dataframe(filename):
    """Lädt das DataFrame aus einer CSV-Datei oder erstellt ein neues, wenn die Datei nicht existiert."""
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['image_file', 'description'])
    return df


def get_png_files(folder_path):
    """Gibt eine Liste von PNG-Dateien im angegebenen Ordner zurück."""
    return glob.glob(f"{folder_path}/*.jpg")


def get_decimal_from_dms(dms, ref):
    """Konvertiert DMS-Koordinaten in Dezimalgrad."""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return degrees + minutes + seconds


def get_exif_location(exif_data):
    """Extrahiere geographische Breite und Länge aus den EXIF-Daten, wenn vorhanden."""
    if not exif_data or 'GPSInfo' not in exif_data:
        return None, None

    gps_info = exif_data['GPSInfo']
    gps_latitude = gps_info.get(2, None)
    gps_latitude_ref = gps_info.get(1, None)
    gps_longitude = gps_info.get(4, None)
    gps_longitude_ref = gps_info.get(3, None)

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
        lon = get_decimal_from_dms(gps_longitude, gps_longitude_ref)
        return lat, lon

    return None, None


def process_image(image_file, df):
    """Verarbeitet ein Bild, extrahiert EXIF-Daten, beschreibt das Bild und aktualisiert das DataFrame."""
    print(f"\nProcessing {image_file}\n")
    with Image.open(image_file) as img:
        exif_data = get_exif_data(img)
        camera_model = exif_data.get('Model', 'Unknown')
        datetime_original = exif_data.get('DateTimeOriginal', 'Unknown')
        latitude, longitude = get_exif_location(exif_data)
        city, country = get_location(latitude, longitude)

        with BytesIO() as buffer:
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()

    full_response = ''
    if city and country:
        prompt = f"Describe the picture taken in {city}, {country} only with bullet points, no sentence or fill words (include text you see in the image):"
    else:
        prompt = "Describe the picture only with bullet points, no sentence or fill words (include text you see in the image):"

    for response in generate(model='llava-phi3:3.8b-mini-fp16', 
                             prompt=prompt, 
                             images=[image_bytes], 
                             stream=True):
        print(response['response'], end='', flush=True)
        full_response += response['response']

    # Speichere die Bildbeschreibung zusammen mit den EXIF-Daten
    new_row_df = pd.DataFrame([{
        'image_file': image_file, 
        'description': full_response, 
        'camera_model': camera_model, 
        'datetime_original': datetime_original,
        'latitude': latitude if latitude is not None else None,
        'longitude': longitude if longitude is not None else None,
        'city': city,
        'country': country
    }])
    
    return pd.concat([df, new_row_df], ignore_index=True)


def main():
    df = load_or_create_dataframe('image_descriptions.csv')
    image_files = get_png_files("./images") 
    image_files.sort()

    print(image_files[:3])
    print(df.head())

    for image_file in image_files:
        if image_file not in df['image_file'].values:
            df = process_image(image_file, df)

    # Save the DataFrame to a CSV file
    df.to_csv('image_descriptions.csv', index=False)


if __name__ == "__main__":
    main()
