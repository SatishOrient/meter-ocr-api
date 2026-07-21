import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from google.cloud import vision

app = Flask(__name__)

API_KEY = 'AIzaSyBqtSFfEf89jKmPWIXQYCNJoR8_nszdJX8'

@app.route('/process-meter', methods=['POST'])
def process_meter():
    try:
        data = request.get_json()
        base64_str = data.get('image', '')

        if not base64_str:
            return jsonify({'reading': 'UNREADABLE'})

        # Base64 to OpenCV Image
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'reading': 'UNREADABLE'})

        # OpenCV Image Enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        _, encoded_img = cv2.imencode('.jpg', thresh)
        content = encoded_img.tobytes()

        client = vision.ImageAnnotatorClient(client_options={"api_key": API_KEY})
        image = vision.Image(content=content)

        response = client.document_text_detection(image=image)
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        # Cleaning & Filtering
        cleaned = full_text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('B', '8')

        import re
        numbers = re.findall(r'\d+', cleaned)

        blocklist = ['2008', '779', '2006', '0101511', '101511', '25', '15', '50']
        reading = 'UNREADABLE'

        for num in numbers:
            if 4 <= len(num) <= 7:
                if num not in blocklist:
                    reading = num
                    break

        return jsonify({'reading': reading})

    except Exception as e:
        return jsonify({'error': str(e), 'reading': 'UNREADABLE'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
