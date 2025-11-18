"""
Streamlit application entry point.
This wrapper allows running the app from the project root using:
    streamlit run app.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the actual app
exec(open("src/ui/app.py", encoding='utf-8').read())
