import cv2
import numpy as np
import onnxruntime as ort
import os
import json
from datetime import datetime

class PaddleOCR:
    def __init__(self, det_model_path, rec_model_path, keys_path):
        self.det_session = ort.InferenceSession(det_model_path)
        self.rec_session = ort.InferenceSession(rec_model_path)
        
        # Load PaddleOCR character keys
        self.load_paddle_keys(keys_path)
        
        print(f"✅ Models loaded successfully")
        print(f"✅ PaddleOCR keys loaded: {len(self.character_list)} characters")

    def load_paddle_keys(self, keys_path):
        """Load PaddleOCR character keys"""
        try:
            with open(keys_path, 'r', encoding='utf-8') as f:
                characters = f.read().strip().split('\n')
            
            # PaddleOCR format: index 0 is blank, 1-N are characters
            self.character_list = ['<blank>'] + characters
            
            print(f"📖 Dictionary sample: {self.character_list[1:11]}")
            print(f"📊 Total characters: {len(self.character_list)} (including blank)")
            
        except Exception as e:
            print(f"❌ Error loading PaddleOCR keys: {e}")
            raise

    def decode_text(self, logits, show_details=False):
        """Decode model output using PaddleOCR dictionary"""
        try:
            # Get predictions
            pred_indices = np.argmax(logits, axis=-1)[0]  # Remove batch dimension
            
            if show_details:
                print(f"📝 Raw predictions: {pred_indices}")
            
            # CTC decoding - remove consecutive duplicates and blanks
            decoded_chars = []
            prev_idx = -1
            
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:  # 0 is blank
                    if idx < len(self.character_list):
                        char = self.character_list[idx]
                        decoded_chars.append(char)
                        if show_details:
                            print(f"  Index {idx} → '{char}' ✓")
                    else:
                        decoded_chars.append(f"[{idx}]")
                        if show_details:
                            print(f"  Index {idx} → [OUT_OF_RANGE] ❌")
                prev_idx = idx
            
            return ''.join(decoded_chars)
            
        except Exception as e:
            print(f"❌ Error in decode_text: {e}")
            return ""

    def test_image(self, image_path, expected_text=None):
        """Test OCR on image"""
        try:
            print(f"\n{'='*60}")
            print(f"🔍 Testing: {os.path.basename(image_path)}")
            if expected_text:
                print(f"📝 Expected: '{expected_text}'")
            print(f"{'='*60}")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot load image: {image_path}")
                return None
            
            print(f"📐 Image size: {image.shape[1]} x {image.shape[0]} pixels")
            
            # Preprocess for recognition
            resized = cv2.resize(image, (320, 48))
            normalized = resized.astype(np.float32) / 255.0
            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb, (2, 0, 1))
            batch = np.expand_dims(chw, axis=0)
            
            # Run recognition
            outputs = self.rec_session.run(None, {"x": batch})
            logits = outputs[0]
            
            print(f"📊 Model output shape: {logits.shape}")
            
            # Decode text
            predicted_text = self.decode_text(logits, show_details=True)
            
            print(f"\n🎯 RESULT:")
            print(f"📝 Predicted: '{predicted_text}'")
            
            if expected_text:
                is_correct = predicted_text == expected_text
                print(f"✅ Correct: {'YES' if is_correct else 'NO'}")
                if not is_correct:
                    print(f"❌ Expected: '{expected_text}'")
                    print(f"❌ Got:      '{predicted_text}'")
            
            return {
                "image": os.path.basename(image_path),
                "predicted": predicted_text,
                "expected": expected_text,
                "correct": predicted_text == expected_text if expected_text else None,
                "image_size": image.shape
            }
            
        except Exception as e:
            print(f"❌ Error testing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_all_images(self):
        """Test all images including the new one"""
        print("🚀 PaddleOCR Complete Test")
        print("="*40)
        
        # Test cases
        test_cases = [
            {
                "path": "/workspaces/ocr/test_image1.png",
                "expected": "别人都在疯狂囤着物资"
            },
            {
                "path": "/workspaces/ocr/test_image2.png", 
                "expected": "别人都在疯狂囤着物资"
            },
            {
                "path": "/workspaces/ocr/image.png",
                "expected": None  # New image, no ground truth
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'🔥' * 5} TEST {i}/{len(test_cases)} {'🔥' * 5}")
            
            result = self.test_image(test_case["path"], test_case["expected"])
            if result:
                results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*60}")
        
        for result in results:
            status = ""
            if result["correct"] is True:
                status = "✅ CORRECT"
            elif result["correct"] is False:
                status = "❌ INCORRECT"
            else:
                status = "🆕 NEW IMAGE"
            
            print(f"🖼️  {result['image']}: '{result['predicted']}' {status}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"/workspaces/ocr/paddle_ocr_results_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_time": datetime.now().isoformat(),
                "dictionary_file": "ppocr_keys_v1.txt",
                "dictionary_size": len(self.character_list),
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {os.path.basename(output_file)}")
        
        return results

def main():
    # File paths
    det_model_path = "/workspaces/ocr/onnx_models/det_model.onnx"
    rec_model_path = "/workspaces/ocr/onnx_models/rec_model.onnx"
    keys_path = "/workspaces/ocr/ppocr_keys_v1.txt"
    
    try:
        # Initialize PaddleOCR
        paddle_ocr = PaddleOCR(det_model_path, rec_model_path, keys_path)
        
        # Test all images
        results = paddle_ocr.test_all_images()
        
        # Final verification
        old_images_correct = sum(1 for r in results[:2] if r.get("correct") == True)
        
        print(f"\n{'='*60}")
        print(f"🎯 VERIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Old images accuracy: {old_images_correct}/2")
        
        if old_images_correct == 2:
            print(f"✅ Old images verified - OCR working correctly!")
            new_result = next((r for r in results if r["image"] == "image.png"), None)
            if new_result:
                print(f"🆕 New image result: '{new_result['predicted']}'")
                print(f"👀 Please manually verify this result with the actual image content.")
        else:
            print(f"❌ Old images failed - need to debug further")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()