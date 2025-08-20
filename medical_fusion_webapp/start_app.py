#!/usr/bin/env python3
"""
Simple startup script for the Medical Image Fusion Web Application
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the app
from app import app, fusion_manager, device

if __name__ == '__main__':
    print("🚀 Medical Image Fusion Web Application")
    print("=" * 50)
    print(f"📱 Device: {device}")
    
    # Display available methods
    available_methods = fusion_manager.get_available_methods()
    categories = fusion_manager.get_method_categories()
    
    print(f"📊 Available fusion methods: {len(available_methods)}")
    for category, methods in categories.items():
        if methods:
            print(f"  {category}: {len(methods)} methods")
            for method_key in methods:
                method_info = fusion_manager.fusion_methods[method_key].get_info()
                status = "✅" if method_info['is_available'] else "❌"
                print(f"    {status} {method_info['name']}")
    
    print("\n🌐 Starting Flask server...")
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)