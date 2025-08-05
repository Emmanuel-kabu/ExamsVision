#!/usr/bin/env python3
"""
Simple camera test script to debug camera access issues
Run this before running the main application to test camera access
"""

import cv2
import time

def test_camera():
    print("Testing camera access...")
    
    # Test different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows default)"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any available backend")
    ]
    
    for backend, name in backends:
        print(f"\n--- Testing {name} ---")
        try:
            cap = cv2.VideoCapture(0, backend)
            
            if not cap.isOpened():
                print(f"❌ Cannot open camera with {name}")
                continue
                
            print(f"✅ Camera opened with {name}")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Successfully read frame: {frame.shape}")
                
                # Test a few more frames
                for i in range(5):
                    ret, frame = cap.read()
                    if ret:
                        print(f"  Frame {i+1}: OK")
                    else:
                        print(f"  Frame {i+1}: Failed")
                        break
                    time.sleep(0.1)
                
                cap.release()
                print(f"✅ {name} is working correctly!")
                return True
            else:
                print(f"❌ Cannot read frames with {name}")
                
            cap.release()
            
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    print("\n❌ No working camera backend found!")
    print("\n💡 Troubleshooting tips:")
    print("1. Check if another application is using the camera")
    print("2. Check Windows camera privacy settings:")
    print("   Settings > Privacy & Security > Camera")
    print("3. Try running this script as administrator")
    print("4. Make sure your camera drivers are installed")
    print("5. Try disconnecting and reconnecting USB camera")
    
    return False

if __name__ == "__main__":
    test_camera()
    input("\nPress Enter to exit...")
