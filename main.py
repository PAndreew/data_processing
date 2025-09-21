import subprocess
import sys
import webbrowser
import time
import os
import socket # For finding a free port

# --- Configuration ---
STREAMLIT_APP_FILE = "app_new.py"
# Port to try for Streamlit. If busy, Streamlit will try the next available.
# We'll try to find an available port and pass it to Streamlit explicitly.
DEFAULT_STREAMLIT_PORT = 8501
STREAMLIT_STARTUP_DELAY = 5 # seconds to wait for Streamlit to start

def find_free_port(start_port):
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            port += 1

def run_streamlit_app():
    streamlit_process = None
    try:
        print("Starting Streamlit app...")

        # Get the directory where the PyInstaller executable is running
        # For development, it's the current script's dir
        # For PyInstaller, it's the temp dir where files are extracted
        if getattr(sys, 'frozen', False):
            # When running as a PyInstaller bundle
            base_path = sys._MEIPASS
        else:
            # When running in a normal Python environment
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Navigate to the Streamlit app's directory (important for relative paths)
        streamlit_app_path = os.path.join(base_path, STREAMLIT_APP_FILE)
        streamlit_dir = os.path.dirname(streamlit_app_path)
        
        # Find a free port
        available_port = find_free_port(DEFAULT_STREAMLIT_PORT)
        streamlit_url = f"http://localhost:{available_port}"

        # Command to run Streamlit
        # We explicitly set the working directory for subprocess
        # Use 'python' or 'python.exe' for the interpreter
        python_executable = sys.executable if not getattr(sys, 'frozen', False) else 'python' # PyInstaller makes a temporary python.exe available
        
        cmd = [
            python_executable, "-m", "streamlit", "run", streamlit_app_path,
            "--server.port", str(available_port),
            "--server.headless", "true", # Don't open browser automatically
            "--server.enableCORS", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Executing command: {' '.join(cmd)}")
        print(f"Changing directory to: {streamlit_dir}")

        # Start Streamlit in a new process, capture output if needed for debugging
        # For a production app, you might redirect stdout/stderr to files or suppress them
        streamlit_process = subprocess.Popen(
            cmd,
            cwd=streamlit_dir, # Ensure Streamlit runs from its own directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit for Streamlit to fully start
        print(f"Waiting {STREAMLIT_STARTUP_DELAY} seconds for Streamlit to start...")
        time.sleep(STREAMLIT_STARTUP_DELAY)

        print(f"Opening browser to {streamlit_url}")
        webbrowser.open_new(streamlit_url)

        # Keep the main process alive. This is crucial for PyInstaller,
        # otherwise the executable would just terminate after opening the browser.
        # A simple input() or a loop watching the Streamlit process works.
        # Or just let Streamlit's process keep this alive.
        # For this method, Streamlit's process is backgrounded, so this script needs to wait.
        print("Streamlit process running in background. Close this console to terminate.")
        
        # A more robust way: monitor the Streamlit process
        # while streamlit_process.poll() is None:
        #     time.sleep(1) # Check every second
        
        # For a simple console app, just keeping it open until user closes is fine.
        # If you want a GUI, you'd integrate this with a GUI main loop.
        # For now, let's keep it simple and expect the user to close the console.
        streamlit_process.wait() # This makes the main script wait until Streamlit process exits

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if streamlit_process and streamlit_process.poll() is None:
            print("Terminating Streamlit process.")
            streamlit_process.terminate()
            streamlit_process.wait() # Wait for it to actually terminate

if __name__ == "__main__":
    run_streamlit_app()