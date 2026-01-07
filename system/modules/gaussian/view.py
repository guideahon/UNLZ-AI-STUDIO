import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import os
import sys
import shutil
from datetime import datetime
from modules.base import StudioModule

class GaussianModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "gaussian", "Gaussian Splatting")
        self.view = None

    def get_view(self) -> ctk.CTkFrame:
        if self.view is None:
            self.view = GaussianView(self.parent)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass

class GaussianView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Main Layout: Sidebar (Left) + Main Content (Right)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Scene Library", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.refresh_button = ctk.CTkButton(self.sidebar_frame, text="Refresh List", command=self.load_scenes)
        self.refresh_button.grid(row=1, column=0, padx=20, pady=10)

        self.scene_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame, label_text="Generated Scenes")
        self.scene_list_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        # --- Main Content ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # Header
        self.header_frame = ctk.CTkFrame(self.main_frame)
        self.header_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="SharpSplat", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=10)
        
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="Create 3D Gaussian Splats for Quest 3", font=ctk.CTkFont(size=14))
        self.subtitle_label.pack(pady=(0, 10))

        # Input Section
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, pady=10, sticky="ew")
        
        self.input_label = ctk.CTkLabel(self.input_frame, text="Input Image or Folder:")
        self.input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.input_entry = ctk.CTkEntry(self.input_frame, width=400, placeholder_text="Select an image or folder...")
        self.input_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.browse_button = ctk.CTkButton(self.input_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.input_frame.grid_columnconfigure(1, weight=1)

        # Actions
        self.action_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.action_frame.grid(row=1, column=0, columnspan=3, pady=10)

        self.process_button = ctk.CTkButton(self.action_frame, text="Generate 3D Splat", command=self.start_processing, fg_color="green", hover_color="darkgreen", height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.process_button.pack(side="left", padx=10)

        # Log Section
        self.log_frame = ctk.CTkFrame(self.main_frame)
        self.log_frame.grid(row=2, column=0, pady=10, sticky="nsew")
        
        self.log_label = ctk.CTkLabel(self.log_frame, text="Process Log:")
        self.log_label.pack(anchor="w", padx=10, pady=(10, 0))
        
        self.log_textbox = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_textbox.configure(state="disabled")

        # Footer / Results
        self.footer_frame = ctk.CTkFrame(self.main_frame)
        self.footer_frame.grid(row=3, column=0, pady=(10, 0), sticky="ew")

        self.open_folder_button = ctk.CTkButton(self.footer_frame, text="Open Output Folder", command=self.open_output_folder, state="disabled")
        self.open_folder_button.pack(side="left", padx=10, pady=10)
        
        self.view_result_button = ctk.CTkButton(self.footer_frame, text="View Result (Browser)", command=self.view_result, state="disabled")
        self.view_result_button.pack(side="right", padx=10, pady=10)

        self.current_output_dir = None
        self.scene_buttons = []
        
        # Start model check in background
        self.after(100, self.start_model_check)
        # Load scenes
        self.after(500, self.load_scenes)

    def load_scenes(self):
        # Clear existing buttons
        for btn in self.scene_buttons:
            btn.destroy()
        self.scene_buttons = []

        # Assuming output is relative to where the script runs, but for a module we might need to be specific
        # We will use the module directory for now or a global user directory.
        # Let's use os.getcwd() for compatibility with existing code structure if run from root
        output_base = os.path.join(os.getcwd(), "modules", "gaussian", "output")
        if not os.path.exists(output_base):
            try:
                os.makedirs(output_base, exist_ok=True)
            except OSError:
                 # Fallback to current dir if permission issues (e.g. Program Files)
                 output_base = os.path.join(os.getcwd(), "output")
                 os.makedirs(output_base, exist_ok=True)
            

        # Find folders starting with splat_
        scenes = []
        try:
            for name in os.listdir(output_base):
                path = os.path.join(output_base, name)
                if os.path.isdir(path) and name.startswith("splat_"):
                    # Parse timestamp for better display
                    try:
                        ts_str = name.replace("splat_", "")
                        dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
                        display_name = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        display_name = name
                    scenes.append((display_name, path))
        except Exception as e:
            self.log(f"Error loading scenes: {e}")

        # Sort by newest first
        scenes.sort(key=lambda x: x[1], reverse=True)

        for name, path in scenes:
            btn = ctk.CTkButton(self.scene_list_frame, text=name, command=lambda p=path: self.select_scene(p), fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
            btn.pack(fill="x", padx=5, pady=2)
            self.scene_buttons.append(btn)

    def select_scene(self, path):
        self.current_output_dir = path
        self.log(f"Selected scene: {os.path.basename(path)}")
        
        # Check if valid
        if os.path.exists(os.path.join(path, "gaussians", "index.html")) or os.path.exists(os.path.join(path, "gaussians", "scene.ply")):
             self.open_folder_button.configure(state="normal")
             self.view_result_button.configure(state="normal")
        else:
             self.log("Warning: Selected scene seems incomplete.")
             self.open_folder_button.configure(state="disabled")
             self.view_result_button.configure(state="disabled")

    def start_model_check(self):
        thread = threading.Thread(target=self.check_model)
        thread.daemon = True
        thread.start()

    def check_model(self):
        # Default model URL and path
        model_url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        model_path = os.path.join(cache_dir, "sharp_2572gikvuh.pt")
        
        self.safe_log("Checking for AI model...")
        
        # Check if file exists and is valid (arbitrary check > 1MB)
        if os.path.exists(model_path):
            try:
                file_size = os.path.getsize(model_path)
                if file_size < 1024 * 1024: # Less than 1MB
                    self.safe_log(f"Found corrupted model file ({file_size} bytes). Deleting...")
                    try:
                        os.remove(model_path)
                    except OSError as e:
                        self.safe_log(f"Error deleting corrupted file: {e}")
                else:
                    self.safe_log(f"Model found at {model_path} ({file_size / 1024 / 1024:.2f} MB).")
                    self.safe_log("Ready to process.")
                    return
            except Exception as e:
                self.safe_log(f"Error checking model file: {e}")

        if not os.path.exists(model_path):
            self.safe_log(f"Model not found or corrupted. Downloading to {model_path}...")
            self.safe_log("Please wait, this may take a minute...")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                import requests
                
                # Use requests with verify=False to bypass SSL errors
                response = requests.get(model_url, stream=True, verify=False)
                response.raise_for_status()
                
                block_size = 1024 * 1024 # 1MB
                downloaded = 0
                
                with open(model_path, 'wb') as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        downloaded += len(data)
                        # Update log sparingly to avoid freezing UI
                        if downloaded % (5 * 1024 * 1024) == 0: # Every 5MB
                            self.safe_log(f"Downloaded {downloaded / 1024 / 1024:.1f} MB...")
                
                self.safe_log("Model downloaded successfully.")
                self.safe_log("Ready to process.")
            except Exception as e:
                self.safe_log(f"Failed to download model: {e}")
                # Try to clean up partial file
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                    except:
                        pass

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)

    def log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def start_processing(self):
        input_path = self.input_entry.get()
        if not input_path or not os.path.exists(input_path):
            self.log("Error: Invalid input path.")
            return
        
        self.process_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.view_result_button.configure(state="disabled")
        
        thread = threading.Thread(target=self.run_process, args=(input_path,))
        thread.start()

    def run_process(self, input_path):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Save to modules/gaussian/output
            output_dir = os.path.join(os.getcwd(), "modules", "gaussian", "output", f"splat_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            self.current_output_dir = output_dir

            self.log(f"Starting processing...")
            self.log(f"Input: {input_path}")
            self.log(f"Output: {output_dir}")

            # Construct command
            # We must assume the environment is set up correctly for 'sharp' to be in path
            # or point to the executable if known. 
            # For now we assume 'sharp' is in PATH or we are in the venv.
            cmd = ["sharp", "predict", "-i", input_path, "-o", output_dir]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            for line in process.stdout:
                self.log(line.strip())

            process.wait()

            if process.returncode == 0:
                self.log("Processing complete!")
                self.setup_viewer(output_dir)
                self.after(0, self.load_scenes)
            else:
                self.log(f"Error: Process exited with code {process.returncode}")

        except Exception as e:
            self.log(f"Exception: {str(e)}")
        finally:
            self.after(0, self.enable_controls)

    def setup_viewer(self, output_dir):
        try:
            # Locate template in the module folder
            viewer_src = os.path.join(os.path.dirname(__file__), "viewer_template.html")
            
            gaussians_dir = os.path.join(output_dir, "gaussians")
            os.makedirs(gaussians_dir, exist_ok=True)
            
            viewer_dst = os.path.join(gaussians_dir, "index.html")
            scene_dst = os.path.join(gaussians_dir, "scene.ply")
            
            # Find the generated .ply file
            ply_files = [f for f in os.listdir(output_dir) if f.endswith(".ply")]
            
            if ply_files:
                # Move the first found ply file to gaussians/scene.ply
                src_ply = os.path.join(output_dir, ply_files[0])
                shutil.move(src_ply, scene_dst)
                self.log(f"Moved {ply_files[0]} to {scene_dst}")
                
                if os.path.exists(viewer_src):
                    shutil.copy(viewer_src, viewer_dst)
                    self.log(f"Viewer created at: {viewer_dst}")
                    self.after(0, lambda: self.open_folder_button.configure(state="normal"))
                    self.after(0, lambda: self.view_result_button.configure(state="normal"))
                else:
                    self.log(f"Error: viewer_template.html not found at {viewer_src}")
            else:
                self.log("Warning: No .ply file found in output.")
        except Exception as e:
            self.log(f"Error setting up viewer: {str(e)}")

    def enable_controls(self):
        self.process_button.configure(state="normal")
        self.browse_button.configure(state="normal")

    def open_output_folder(self):
        if self.current_output_dir:
            os.startfile(os.path.join(self.current_output_dir, "gaussians"))

    def view_result(self):
        if self.current_output_dir:
            # Start a simple HTTP server to avoid CORS issues with file:///
            # We serve the output directory
            port = 8000
            # Find a free port
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while port < 8100:
                result = sock.connect_ex(('127.0.0.1', port))
                if result != 0: # Port is free
                    break
                port += 1
            sock.close()

            def start_server():
                try:
                    # Use python.exe instead of pythonw.exe for the server to avoid console issues
                    # and redirect output to DEVNULL
                    python_exe = sys.executable.replace("pythonw.exe", "python.exe")
                    
                    # Serve the specific output directory
                    handler = subprocess.Popen(
                        [python_exe, "-m", "http.server", str(port), "--directory", self.current_output_dir],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                except Exception as e:
                    print(f"Failed to start server: {e}")

            threading.Thread(target=start_server, daemon=True).start()
            
            # Open browser
            url = f"http://localhost:{port}/gaussians/index.html"
            self.log(f"Opening viewer at {url}")
            import webbrowser
            webbrowser.open(url)
