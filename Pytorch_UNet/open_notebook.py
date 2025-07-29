#!/usr/bin/env python3
"""
Script to open the barnacle notebook in different ways
"""

import os
import subprocess
import sys

def open_notebook_methods():
    print("Barnacle Notebook Opening Options")
    print("=" * 40)
    
    notebook_path = "notebooks/barnacle_unet.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"Notebook not found at {notebook_path}")
        return
    
    print("1. Try opening with nbconvert (HTML):")
    try:
        subprocess.run([
            "jupyter", "nbconvert", "--to", "html", 
            "--output-dir", "notebooks", 
            notebook_path
        ], check=True)
        html_path = notebook_path.replace('.ipynb', '.html')
        if os.path.exists(html_path):
            print(f"   HTML version created: {html_path}")
            print(f"   Open this file in your browser")
        else:
            print("   Failed to create HTML")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Try opening with VS Code:")
    try:
        subprocess.run(["code", notebook_path], check=True)
        print("   VS Code should open the notebook")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Try opening with Jupyter Lab:")
    try:
        subprocess.run(["jupyter", "lab", notebook_path], check=True)
        print("   Jupyter Lab should open")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Alternative: Run the counting demo script:")
    demo_script = "barnacle_counting_demo.py"
    if os.path.exists(demo_script):
        print(f"   Run: python {demo_script}")
        print("   This demonstrates the counting functionality")
    else:
        print(f"   Demo script not found: {demo_script}")

if __name__ == "__main__":
    open_notebook_methods() 