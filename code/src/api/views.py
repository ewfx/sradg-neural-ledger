from django.http import JsonResponse
from rest_framework.decorators import api_view
from data.file_handler import process_uploaded_file
from anomaly_detection.anomaly_detector import detect_anomalies
import json
import logging

# Load configuration dynamically
def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

config = load_config()
logging.basicConfig(filename='logs/anomaly_logs.log', level=logging.INFO)

@api_view(['POST'])
def upload_file(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    case_name = request.POST.get('case_name')
    
    processed_data = process_uploaded_file(uploaded_file)
    anomalies = detect_anomalies(processed_data, config, case_name)
    
    return JsonResponse({'anomalies': anomalies})
