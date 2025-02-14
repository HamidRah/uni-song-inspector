from rest_framework import serializers

class InstrumentDetectionSerializer(serializers.Serializer):
    audio_file = serializers.FileField()