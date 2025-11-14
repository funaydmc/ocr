#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Demo Script
Simple demonstration of Chinese OCR using PaddleOCR models
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
from datetime import datetime

class SimpleOCR:
    def __init__(self):
        """Initialize OCR with default model paths"""
        self.det_model_path = "onnx_models/det_model.onnx"
        self.rec_model_path = "onnx_models/rec_model.onnx" 
        self.keys_path = "ppocr_keys_v1.txt"
        
        self._load_models()
        self._load_characters()

    def _load_models(self):
        """Load ONNX models"""
        try:
            self.det_session = ort.InferenceSession(self.det_model_path)
            self.rec_session = ort.InferenceSession(self.rec_model_path)
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            sys.exit(1)

    def _load_characters(self):
        """Load character dictionary"""
        try:
            with open(self.keys_path, 'r', encoding='utf-8') as f:
                characters = f.read().strip().split('\n')
            self.characters = ['<blank>'] + characters
            print(f"✅ Dictionary loaded: {len(self.characters)} characters")
        except Exception as e:
            print(f"❌ Error loading dictionary: {e}")
            sys.exit(1)

    def recognize_text(self, image_path):
        """Recognize text from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return f"Error: Cannot load image {image_path}"
            
            # Preprocess
            resized = cv2.resize(image, (320, 48))
            normalized = resized.astype(np.float32) / 255.0
            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb, (2, 0, 1))
            batch = np.expand_dims(chw, axis=0)
            
            # Run recognition
            outputs = self.rec_session.run(None, {"x": batch})
            logits = outputs[0]
            
            # Decode
            pred_indices = np.argmax(logits, axis=-1)[0]
            
            # CTC decode
            decoded_chars = []
            prev_idx = -1
            
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:
                    if idx < len(self.characters):
                        decoded_chars.append(self.characters[idx])
                prev_idx = idx
            
            return ''.join(decoded_chars)
            
        except Exception as e:
            return f"Error processing {image_path}: {e}"

def main():
    print("🚀 Simple OCR Demo")
    print("=" * 30)
    
    # Initialize OCR
    ocr = SimpleOCR()
    
    # Test images
    test_images = [
        "test_image1.png",
        "test_image2.png", 
        "image.png"
    ]
    
    print("\n📝 OCR Results:")
    print("-" * 40)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            result = ocr.recognize_text(image_path)
            print(f"{image_path:15} → {result}")
        else:
            print(f"{image_path:15} → [File not found]")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()