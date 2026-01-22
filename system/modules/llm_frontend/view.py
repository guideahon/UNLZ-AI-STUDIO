import customtkinter as ctk
import os
import threading
import requests
import json
from pathlib import Path
from modules.base import StudioModule
from tkinter import filedialog, messagebox, simpledialog

class LLMFrontendModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "llm_frontend", "Chat y Gestor LLM")
        self.view = None
        self.profile_manager = parent.profile_manager
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        # Pass main app manager to control local server
        self.view = LLMFrontendView(self.app.main_container, self.profile_manager, self.app)
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
        self.manager = app.manager
        
        # Dedicated config for Chat Server
        self.chat_port = 8081
        self.api_url = f"http://127.0.0.1:{self.chat_port}/v1/chat/completions"
        self.loaded_model = None
        
        tr = self.app.tr
        
        # Layout: Tabs for Chat, Models, Search
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_chat = self.tabview.add(tr("tab_chat"))
        self.tab_models = self.tabview.add(tr("tab_models"))
        self.tab_search = self.tabview.add(tr("tab_download"))
        
        self.build_chat_tab()
        self.build_models_tab()
        self.build_search_tab()
        
        # Initial model scan
        self.refresh_models()

    # --- Chat Tab ---
    def build_chat_tab(self):
        tr = self.app.tr
        frame = self.tab_chat
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Header: Server Status & Model
        header = ctk.CTkFrame(frame, height=50, fg_color=("gray95", "#2A2A2A"))
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(header, text=tr("lbl_server_port"), font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        self.status_indicator = ctk.CTkLabel(
            header,
            text=tr("status_stopped"),
            text_color="red",
            image=self.app.get_status_matrix_image(),
            compound="left",
        )
        self.status_indicator.pack(side="left")
        
        ctk.CTkLabel(header, text=tr("lbl_active_model"), font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(20, 5))
        self.model_selector = ctk.CTkComboBox(header, width=300, values=[tr("placeholder_select_model")])
        self.model_selector.pack(side="left")
        
        self.btn_load = ctk.CTkButton(header, text=tr("btn_load"), width=80, command=self.toggle_chat_server)
        self.btn_load.pack(side="left", padx=10)
        
        # Chat Area
        self.chat_history = ctk.CTkTextbox(frame, state="disabled", font=ctk.CTkFont(size=14))
        self.chat_history.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Input
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text=tr("chat_input_placeholder"), height=40)
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.chat_input.bind("<Return>", self.send_message)
        
        self.send_btn = ctk.CTkButton(input_frame, text=tr("btn_send"), command=self.send_message, width=100, height=40)
        self.send_btn.pack(side="right")
        
        # Status
        self.chat_status = ctk.CTkLabel(frame, text=tr("status_ready"), text_color="gray")
        self.chat_status.grid(row=3, column=0, sticky="e", padx=10)
        
        # Check initial state
        self.check_server_state()

    def check_server_state(self):
        if self.manager.is_running("llm_chat"):
            self.status_indicator.configure(text=self.app.tr("status_running"), text_color="green")
            self.btn_load.configure(text=self.app.tr("btn_stop"), fg_color="red")
            self.chat_input.configure(state="normal")
            self.send_btn.configure(state="normal")
        else:
            self.status_indicator.configure(text=self.app.tr("status_stopped"), text_color="red")
            self.btn_load.configure(text=self.app.tr("btn_load"), fg_color="green")
            # We don't disable input to allow typing, but send will fail
            
        self.after(2000, self.check_server_state)

    def toggle_chat_server(self):
        if self.manager.is_running("llm_chat"):
            self.manager.stop("llm_chat")
            self.check_server_state()
            return

        model_name = self.model_selector.get()
        if not model_name or model_name == self.app.tr("placeholder_select_model"):
            messagebox.showwarning("Model Required", self.app.tr("msg_model_select_warning"))
            return

        # Find full path
        model_path = None
        search_dir = Path(os.environ.get("LLAMA_MODEL_DIR", r"C:\models"))
        for f in search_dir.rglob("*.gguf"):
            if f.name == model_name:
                model_path = f
                break
        
        if not model_path:
             messagebox.showerror("Error", self.app.tr("msg_model_file_not_found"))
             return

        self.btn_load.configure(state="disabled", text=self.app.tr("status_starting"))
        threading.Thread(target=self._start_server_bg, args=(model_path,)).start()

    def _start_server_bg(self, model_path):
        try:
            config = {
                "model_path": str(model_path),
                "port": self.chat_port,
                "host": "127.0.0.1",
                "ctx_size": 2048, # Simple default
                "n_gpu_layers": 99 # Try max
            }
            self.manager.start_process("llm_chat", config)
            self.after(0, self.check_server_state)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Start Failed", str(e)))
            self.after(0, self.check_server_state)

    def send_message(self, event=None):
        text = self.chat_input.get()
        if not text.strip(): return
        
        if not self.manager.is_running("llm_chat"):
            messagebox.showwarning("Server Stopped", self.app.tr("msg_server_stopped_warning"))
            return

        self.append_chat(self.app.tr("role_user"), text)
        self.chat_input.delete(0, "end")
        self.chat_status.configure(text=self.app.tr("status_thinking"), text_color="#EAB308")
        
        threading.Thread(target=self._run_inference, args=(text,), daemon=True).start()

    def _run_inference(self, prompt):
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.after(0, lambda: self.append_chat(self.app.tr("role_ai"), answer))
            self.after(0, lambda: self.chat_status.configure(text=self.app.tr("status_ready"), text_color="gray"))
            
        except Exception as e:
            self.after(0, lambda: self.append_chat(self.app.tr("role_system"), f"Error: {e}"))
            self.after(0, lambda: self.chat_status.configure(text=self.app.tr("status_error"), text_color="red"))

    def append_chat(self, role, text):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"\n[{role}]\n", role)
        self.chat_history.insert("end", f"{text}\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")

    # --- Models Tab ---
    def build_models_tab(self):
        tr = self.app.tr
        frame = self.tab_models
        
        ctk.CTkLabel(frame, text=tr("lbl_model_library"), font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        dir_frame = ctk.CTkFrame(frame, fg_color="transparent")
        dir_frame.pack(fill="x", padx=10)
        
        default_dir = self.app.get_setting("model_dir", os.environ.get("LLAMA_MODEL_DIR", r"C:\models"))
        self.model_dir_var = ctk.StringVar(value=default_dir)
        ctk.CTkEntry(dir_frame, textvariable=self.model_dir_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(dir_frame, text=tr("btn_browse"), width=80, command=self.browse_model_dir).pack(side="right")
        
        ctk.CTkButton(frame, text=tr("btn_refresh_list"), command=self.refresh_models).pack(pady=10)

        self.models_scroll = ctk.CTkScrollableFrame(frame, label_text=tr("lbl_files"))
        self.models_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
    def browse_model_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.model_dir_var.set(d)
            self.app.set_setting("model_dir", d)
            self.refresh_models()

    def refresh_models(self):
        tr = self.app.tr
        for widget in self.models_scroll.winfo_children():
            widget.destroy()
            
        path = Path(self.model_dir_var.get())
        self.app.set_setting("model_dir", str(path))
        if not path.exists():
            return

        files = list(path.rglob("*.gguf"))
        model_names = [f.name for f in files]
        
        # Update dropdown
        if model_names:
            self.model_selector.configure(values=model_names)
            self.model_selector.set(model_names[0])
        else:
            self.model_selector.configure(values=[tr("placeholder_select_model")])
            self.model_selector.set(tr("placeholder_select_model"))
            
        for f in files:
            card = ctk.CTkFrame(self.models_scroll, fg_color=("gray90", "#2A2A2A"))
            card.pack(fill="x", pady=2)
            ctk.CTkLabel(card, text=f.name).pack(side="left", padx=10)
            
            # Delete Button
            del_btn = ctk.CTkButton(card, text=tr("btn_delete"), width=60, fg_color="#EF4444", hover_color="#DC2626",
                                    command=lambda p=f: self.delete_model(p))
            del_btn.pack(side="right", padx=10, pady=5)

    def delete_model(self, file_path: Path):
        """Confirm and delete the selected model file."""
        if not messagebox.askyesno(self.app.tr("title_delete_model"), self.app.tr("msg_delete_confirm").format(file_path.name)):
            return
            
        try:
            os.remove(file_path)
            self.refresh_models()
            messagebox.showinfo("Deleted", self.app.tr("msg_deleted_success").format(file_path.name))
        except Exception as e:
            messagebox.showerror("Error", self.app.tr("msg_delete_error").format(str(e)))

    # --- Search / Download Tab ---
    def build_search_tab(self):
        tr = self.app.tr
        frame = self.tab_search
        
        ctk.CTkLabel(frame, text=tr("lbl_download_hf"), font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(padx=20, pady=10)
        
        ctk.CTkLabel(grid, text=tr("lbl_repo_id")).grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.dl_repo = ctk.CTkEntry(grid, width=300)
        self.dl_repo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(grid, text=tr("lbl_filename")).grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.dl_filename = ctk.CTkEntry(grid, width=300)
        self.dl_filename.grid(row=1, column=1, padx=5, pady=5)
        
        self.dl_status = ctk.CTkLabel(frame, text="", text_color="#EAB308")
        self.dl_status.pack(pady=5)
        
        ctk.CTkButton(frame, text=tr("btn_download"), command=self.start_download).pack(pady=20)
        
        # Popular Models Section
        ctk.CTkLabel(frame, text=tr("lbl_popular_models"), font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        preset_frame = ctk.CTkScrollableFrame(frame, height=200, label_text=tr("lbl_click_to_fill"))
        preset_frame.pack(fill="x", padx=10, pady=5)
        
        presets = [
            ("Llama 3.2 3B (Instruct)", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            ("Qwen 2.5 7B (Instruct)", "Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"),
            ("Qwen 2.5 14B (Instruct)", "Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf"),
            ("GLM 4.7 Flash", "unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
            ("Mistral 7B v0.3", "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
            ("DeepSeek R1 Distill (8B)", "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
            ("Phi-3.5 Mini", "microsoft/Phi-3.5-mini-instruct-gguf", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
            ("Hermes 3 Llama 3.1 8B", "NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"),
        ]
        
        for name, repo, file in presets:
            btn = ctk.CTkButton(preset_frame, text=name, 
                                command=lambda r=repo, f=file: self.fill_download(r, f),
                                fg_color="transparent", border_width=1, text_color=("gray10", "gray90"))
            btn.pack(fill="x", pady=2)

        ctk.CTkLabel(frame, text=tr("lbl_download_note"), text_color="gray").pack(side="bottom", pady=20)

    def fill_download(self, repo, filename):
        self.dl_repo.delete(0, "end")
        self.dl_repo.insert(0, repo)
        self.dl_filename.delete(0, "end")
        self.dl_filename.insert(0, filename)
        self.dl_status.configure(text=f"Selected: {filename}")

    def start_download(self):
        repo = self.dl_repo.get().strip()
        filename = self.dl_filename.get().strip()
        
        if not repo or not filename:
            self.dl_status.configure(text=self.app.tr("msg_fill_fields"))
            return
            
        self.dl_status.configure(text=self.app.tr("status_download_progress"))
        threading.Thread(target=self._download_worker, args=(repo, filename), daemon=True).start()

    def _download_worker(self, repo, filename):
        try:
            from huggingface_hub import hf_hub_download
            dest_dir = Path(self.model_dir_var.get())
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir, local_dir_use_symlinks=False)
            self.after(0, lambda: self.dl_status.configure(text=self.app.tr("status_download_done").format(Path(path).name)))
            self.after(0, lambda: messagebox.showinfo("Download Complete", self.app.tr("msg_download_complete").format(filename)))
            self.after(0, self.refresh_models)
        except Exception as e:
            self.after(0, lambda: self.dl_status.configure(text=self.app.tr("status_download_error").format(str(e))))
