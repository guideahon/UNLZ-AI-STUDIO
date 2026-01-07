import customtkinter as ctk
import os
import threading
import requests
import json
from pathlib import Path
from modules.base import StudioModule
from tkinter import filedialog, messagebox

class LLMFrontendModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "llm_frontend", "LLM Frontend")
        self.view = None
        self.profile_manager = parent.profile_manager # Access main app's profile manager
        self.app = parent # Used to access tr()

    def get_view(self) -> ctk.CTkFrame:
        # Re-create view to ensure translations match current language
        self.view = LLMFrontendView(self.parent, self.profile_manager, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass

class LLMFrontendView(ctk.CTkFrame):
    def __init__(self, master, profile_manager, app, **kwargs):
        super().__init__(master, **kwargs)
        self.profile_manager = profile_manager
        self.app = app
        
        # Helper for shorter code
        tr = self.app.tr
        
        # Layout: Tabs for Chat, Models, Search
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_chat = self.tabview.add(tr("tab_chat"))
        self.tab_models = self.tabview.add(tr("tab_models"))
        self.tab_search = self.tabview.add(tr("tab_download"))
        self.tab_params = self.tabview.add(tr("tab_params"))

        self.build_chat_tab()
        self.build_models_tab()
        self.build_search_tab()
        self.build_params_tab()

    # --- Chat Tab ---
    def build_chat_tab(self):
        tr = self.app.tr
        frame = self.tab_chat
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        
        self.chat_history = ctk.CTkTextbox(frame, state="disabled", font=ctk.CTkFont(size=14))
        self.chat_history.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 10))
        
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text=tr("chat_input_placeholder"), height=40)
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.chat_input.bind("<Return>", self.send_message)
        
        self.send_btn = ctk.CTkButton(input_frame, text=tr("btn_send"), command=self.send_message, width=100, height=40)
        self.send_btn.pack(side="right")
        
        # Status
        self.chat_status = ctk.CTkLabel(frame, text="Ready", text_color="gray")
        self.chat_status.grid(row=2, column=0, sticky="e", padx=10)

    def send_message(self, event=None):
        text = self.chat_input.get()
        if not text.strip(): return
        
        self.append_chat("User", text)
        self.chat_input.delete(0, "end")
        self.chat_status.configure(text="Thinking...", text_color="#EAB308")
        
        threading.Thread(target=self._run_inference, args=(text,), daemon=True).start()

    def _run_inference(self, prompt):
        try:
            # Assumes gateway running on default port or env
            # We can try to guess from env or just localhost:8000 (gateway default)
            port = "8000" 
            url = f"http://127.0.0.1:{port}/llm" 
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            # Streaming could be harder to implement quickly, let's do standard post first
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.after(0, lambda: self.append_chat("AI", answer))
            self.after(0, lambda: self.chat_status.configure(text="Ready", text_color="gray"))
            
        except Exception as e:
            self.after(0, lambda: self.append_chat("System", f"Error: {e}"))
            self.after(0, lambda: self.chat_status.configure(text="Error", text_color="red"))

    def append_chat(self, role, text):
        self.chat_history.configure(state="normal")
        tag = "user" if role == "User" else "ai"
        color = "#EAB308" if role == "User" else "#3B82F6"
        if role == "System": color = "red"
        
        self.chat_history.insert("end", f"\n[{role}]\n", role)
        self.chat_history.insert("end", f"{text}\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")

    # --- Models Tab ---
    def build_models_tab(self):
        tr = self.app.tr
        frame = self.tab_models
        
        ctk.CTkLabel(frame, text="Available GGUF Models", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Directory selection
        dir_frame = ctk.CTkFrame(frame, fg_color="transparent")
        dir_frame.pack(fill="x", padx=10)
        
        self.model_dir_var = ctk.StringVar(value=os.environ.get("LLAMA_MODEL_DIR", r"C:\models"))
        ctk.CTkEntry(dir_frame, textvariable=self.model_dir_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(dir_frame, text="Browse", width=80, command=self.browse_model_dir).pack(side="right")
        
        ctk.CTkButton(frame, text="Refresh List", command=self.refresh_models).pack(pady=10)

        self.models_scroll = ctk.CTkScrollableFrame(frame, label_text="Files")
        self.models_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.refresh_models()

    def browse_model_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.model_dir_var.set(d)
            self.refresh_models()

    def refresh_models(self):
        tr = self.app.tr
        for widget in self.models_scroll.winfo_children():
            widget.destroy()
            
        path = Path(self.model_dir_var.get())
        if not path.exists():
            ctk.CTkLabel(self.models_scroll, text="Directory not found").pack()
            return

        try:
            files = list(path.rglob("*.gguf"))
            if not files:
                ctk.CTkLabel(self.models_scroll, text="No .gguf files found").pack()
            
            for f in files:
                card = ctk.CTkFrame(self.models_scroll, fg_color=("gray90", "#2A2A2A"))
                card.pack(fill="x", pady=2)
                ctk.CTkLabel(card, text=f.name).pack(side="left", padx=10)
                ctk.CTkButton(card, text=tr("btn_load"), width=60, command=lambda p=str(f): self.load_model(p)).pack(side="right", padx=10, pady=5)
        except Exception as e:
            ctk.CTkLabel(self.models_scroll, text=f"Error: {e}").pack()

    def load_model(self, path):
        # Update runtime profile to use this model
        # We assume 'personalizado' profile allows custom model
        ok, msg = self.profile_manager.set_active_profile("personalizado")
        if not ok:
            messagebox.showerror("Error", msg)
            return

        updates = {"model_path": path}
        ok, msg = self.profile_manager.update_custom_settings(updates)
        if ok:
            messagebox.showinfo("Success", f"Model selected: {Path(path).name}\nPlease restart Server in Monitor tab.")
        else:
            messagebox.showerror("Error", msg)

    # --- Search / Download Tab ---
    def build_search_tab(self):
        tr = self.app.tr
        frame = self.tab_search
        
        ctk.CTkLabel(frame, text="Download from HuggingFace", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(padx=20, pady=10)
        
        ctk.CTkLabel(grid, text="Repo ID (e.g. TheBloke/Llama-2-7b-Chat-GGUF):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.dl_repo = ctk.CTkEntry(grid, width=300)
        self.dl_repo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(grid, text="Filename (e.g. llama-2-7b-chat.Q4_K_M.gguf):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.dl_filename = ctk.CTkEntry(grid, width=300)
        self.dl_filename.grid(row=1, column=1, padx=5, pady=5)
        
        self.dl_status = ctk.CTkLabel(frame, text="", text_color="#EAB308")
        self.dl_status.pack(pady=5)
        
        ctk.CTkButton(frame, text="Download", command=self.start_download).pack(pady=20)
        
        ctk.CTkLabel(frame, text="Note: Files are downloaded to 'models/' folder.", text_color="gray").pack(side="bottom", pady=20)

    def start_download(self):
        repo = self.dl_repo.get().strip()
        filename = self.dl_filename.get().strip()
        
        if not repo or not filename:
            self.dl_status.configure(text="Please fill both fields")
            return
            
        self.dl_status.configure(text="Downloading... check console for progress")
        threading.Thread(target=self._download_worker, args=(repo, filename), daemon=True).start()

    def _download_worker(self, repo, filename):
        try:
            from huggingface_hub import hf_hub_download
            
            # Destination: C:\models or local
            dest_dir = Path(self.model_dir_var.get())
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            ctk.CTkLabel(self.tab_search, text="Downloading...").pack()
            
            path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir, local_dir_use_symlinks=False)
            
            self.after(0, lambda: self.dl_status.configure(text=f"Done! Saved to {Path(path).name}"))
            self.after(0, lambda: messagebox.showinfo("Download Complete", f"Downloaded {filename}"))
            self.after(0, self.refresh_models)
            
        except Exception as e:
            self.after(0, lambda: self.dl_status.configure(text=f"Error: {str(e)}"))

    # --- Params Tab ---
    def build_params_tab(self):
        tr = self.app.tr
        frame = self.tab_params
        
        # Simple sliders for common params
        self.params_vars = {}
        
        settings = self.profile_manager.get_custom_settings()
        
        params = [
            ("temperature", "Temperature", 0.0, 2.0, 0.1),
            ("top_p", "Top P", 0.0, 1.0, 0.05),
            ("ctx_size", "Context Size", 512, 8192, 256),
            ("n_gpu_layers", "GPU Layers", 0, 100, 1),
        ]
        
        for i, (key, label, vmin, vmax, step) in enumerate(params):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=10)
            
            val = settings.get(key, vmin)
            var = ctk.DoubleVar(value=float(val))
            self.params_vars[key] = var
            
            ctk.CTkLabel(row, text=label, width=100).pack(side="left")
            slider = ctk.CTkSlider(row, from_=vmin, to=vmax, number_of_steps=(vmax-vmin)/step, variable=var)
            slider.pack(side="left", fill="x", expand=True, padx=10)
            ctk.CTkLabel(row, textvariable=var, width=50).pack(side="right")

        ctk.CTkButton(frame, text="Save Parameters", command=self.save_params).pack(pady=20)

    def save_params(self):
        updates = {k: v.get() for k, v in self.params_vars.items()}
        # Int conversion for some
        updates["ctx_size"] = int(updates["ctx_size"])
        updates["n_gpu_layers"] = int(updates["n_gpu_layers"])
        
        ok, msg = self.profile_manager.update_custom_settings(updates)
        if ok:
            messagebox.showinfo("Saved", "Parameters saved to Custom profile.")
        else:
            messagebox.showerror("Error", msg)
