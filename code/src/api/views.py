import pandas as pd
from rest_framework.response import Response
from rest_framework.decorators import api_view
from src.anomaly_detection.detector import detect_anomalies

@api_view(['POST'])
def upload_csv(request):
    file = request.FILES.get('file')
    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    df = pd.read_csv(file)
    anomalies = detect_anomalies(df)
    return Response({"anomalies": anomalies})
