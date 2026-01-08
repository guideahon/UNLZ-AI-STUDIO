import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import sys
import threading
import subprocess
import webbrowser
import re
from datetime import datetime
from pathlib import Path

from modules.base import StudioModule


class Model3DModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "model_3d", "Generacion de modelo 3D")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = Model3DView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#222222",
            foreground="white",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
        )
        label.pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class Model3DView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr

        self.app_root = Path(__file__).resolve().parents[4]
        self.data_dir = Path(__file__).resolve().parents[3] / "data" / "model_3d"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.data_dir / "config.json"

        self.output_base = self.app_root / "system" / "3d-out"
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.config = self.load_config()
        self.backends = [
            ("hunyuan3d2", self.tr("model3d_backend_hunyuan")),
            ("reconv", self.tr("model3d_backend_reconv")),
            ("sam3d", self.tr("model3d_backend_sam3d")),
            ("stepx1", self.tr("model3d_backend_stepx1")),
        ]
        self.backend_links = {
            "hunyuan3d2": "https://www.youtube.com/watch?v=zqTTYYqHBhc",
            "reconv": "https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen",
            "sam3d": "https://github.com/facebookresearch/sam-3d-objects",
            "stepx1": "https://github.com/stepfun-ai/Step1X-3D",
        }
        self.backend_spaces = {
            "reconv": "https://huggingface.co/spaces/Stable-X/ReconViaGen",
            "hunyuan3d2": "https://huggingface.co/spaces/tencent/Hunyuan3D-2",
            "sam3d": "https://ai.meta.com/sam3d/",
            "stepx1": "https://huggingface.co/spaces/stepfun-ai/Step1X-3D",
        }

        self.backend_key = self.config.get("backend", "hunyuan3d2")
        self.backend_label_var = ctk.StringVar(value=self.label_for_key(self.backend_key))

        self.input_mode_labels = {
            "single": self.tr("model3d_input_mode_single"),
            "multi": self.tr("model3d_input_mode_multi"),
            "video": self.tr("model3d_input_mode_video"),
        }
        self.input_mode_var = ctk.StringVar(value=self.input_mode_labels["single"])
        self.input_paths = []

        self.weight_options = []
        self.weight_var = ctk.StringVar(value="")
        self._task_running = False
        self.hf_token_var = ctk.StringVar()
        self.last_output_dir = self.config.get("last_output_dir")

        self.build_ui()
        self.refresh_hf_token_ui()
        self.update_backend_ui()
        self.update_weights_menu()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("model3d_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("model3d_subtitle"), text_color="gray").pack(anchor="w")

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill="x", padx=10, pady=(10, 5))
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text=self.tr("model3d_input_mode_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        input_values = [self.input_mode_labels[key] for key in ("single", "multi", "video")]
        self.input_mode_menu = ctk.CTkOptionMenu(input_frame, values=input_values, variable=self.input_mode_var)
        self.input_mode_menu.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 5))

        self.select_files_button = ctk.CTkButton(input_frame, text=self.tr("model3d_btn_select_files"), command=self.select_inputs, width=180)
        self.select_files_button.grid(row=0, column=2, sticky="e", padx=10, pady=(10, 5))

        ctk.CTkLabel(input_frame, text=self.tr("model3d_selected_label")).grid(row=1, column=0, sticky="nw", padx=10, pady=(0, 10))
        self.selected_box = ctk.CTkTextbox(input_frame, height=90)
        self.selected_box.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.selected_box.configure(state="disabled")

        settings = ctk.CTkFrame(self)
        settings.pack(fill="x", padx=10, pady=(10, 5))
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text=self.tr("model3d_backend_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        backend_values = [label for _, label in self.backends]
        self.backend_menu = ctk.CTkOptionMenu(settings, values=backend_values, variable=self.backend_label_var, command=self.on_backend_change)
        self.backend_menu.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 5))

        self.open_backend_button = ctk.CTkButton(settings, text=self.tr("model3d_btn_open_backend"), command=self.open_backend_link, width=160)
        self.open_backend_button.grid(row=0, column=2, sticky="e", padx=10, pady=(10, 5))

        ctk.CTkLabel(settings, text=self.tr("model3d_actions_label")).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        actions = ctk.CTkFrame(settings, fg_color="transparent")
        actions.grid(row=1, column=1, columnspan=2, sticky="ew", padx=10, pady=5)

        self.install_backend_button = ctk.CTkButton(actions, text=self.tr("model3d_btn_install_backend"), command=self.install_backend_action)
        self.install_backend_button.pack(side="left", padx=5)
        self.uninstall_backend_button = ctk.CTkButton(actions, text=self.tr("model3d_btn_uninstall_backend"), command=self.uninstall_backend_action)
        self.uninstall_backend_button.pack(side="left", padx=5)
        self.install_weights_button = ctk.CTkButton(actions, text=self.tr("model3d_btn_install_weights"), command=self.install_weights_action)
        self.install_weights_button.pack(side="left", padx=5)
        self.uninstall_weights_button = ctk.CTkButton(actions, text=self.tr("model3d_btn_uninstall_weights"), command=self.uninstall_weights_action)
        self.uninstall_weights_button.pack(side="left", padx=5)

        ctk.CTkLabel(settings, text=self.tr("model3d_weights_label")).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.weights_menu = ctk.CTkOptionMenu(settings, values=[], variable=self.weight_var, command=self.on_weight_change)
        self.weights_menu.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        self.open_hf_button = ctk.CTkButton(settings, text=self.tr("model3d_btn_open_hf"), command=self.open_hf_access, width=160)
        self.open_hf_button.grid(row=2, column=2, sticky="e", padx=10, pady=5)

        self.hf_token_label = ctk.CTkLabel(settings, text=self.tr("model3d_hf_token_label"))
        self.hf_token_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.hf_token_entry = ctk.CTkEntry(settings, textvariable=self.hf_token_var, width=480, show="*")
        self.hf_token_entry.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        self.hf_token_button = ctk.CTkButton(settings, text=self.tr("model3d_btn_save_hf"), command=self.save_hf_token, width=160)
        self.hf_token_button.grid(row=3, column=2, sticky="e", padx=10, pady=5)

        self.hf_token_link = ctk.CTkButton(settings, text=self.tr("model3d_btn_open_hf_tokens"), command=self.open_hf_tokens, width=160)
        self.hf_token_link.grid(row=4, column=2, sticky="e", padx=10, pady=(0, 5))
        ToolTip(self.hf_token_link, self.tr("model3d_hf_token_perms"))

        self.status_label = ctk.CTkLabel(settings, text=self.tr("model3d_status_idle"), text_color="gray")
        self.status_label.grid(row=4, column=1, sticky="w", padx=10, pady=(0, 5))

        output_frame = ctk.CTkFrame(self)
        output_frame.pack(fill="x", padx=10, pady=(10, 5))
        output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(output_frame, text=self.tr("model3d_output_label")).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        self.output_entry = ctk.CTkEntry(output_frame, width=480, placeholder_text=self.tr("model3d_output_placeholder"))
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        ctk.CTkButton(output_frame, text=self.tr("btn_browse"), command=self.browse_output, width=100).grid(row=0, column=2, padx=10, pady=10)

        actions_footer = ctk.CTkFrame(self, fg_color="transparent")
        actions_footer.pack(fill="x", padx=10, pady=(10, 5))

        self.generate_button = ctk.CTkButton(actions_footer, text=self.tr("model3d_btn_generate"), command=self.start_generate, height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.generate_button.pack(side="left", padx=5)

        self.open_output_button = ctk.CTkButton(actions_footer, text=self.tr("model3d_btn_open_output"), command=self.open_output_folder, state="disabled")
        self.open_output_button.pack(side="left", padx=5)

        self.open_generados_button = ctk.CTkButton(actions_footer, text=self.tr("model3d_btn_open_generados"), command=self.open_generados_folder)
        self.open_generados_button.pack(side="left", padx=5)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text=self.tr("model3d_log_label")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.log_box.configure(state="disabled")

        terminal = ctk.CTkFrame(self)
        terminal.pack(fill="x", padx=10, pady=(0, 10))
        terminal.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(terminal, text=self.tr("model3d_py_terminal_label")).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.py_cmd_entry = ctk.CTkEntry(terminal, placeholder_text=self.tr("model3d_py_terminal_placeholder"))
        self.py_cmd_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(terminal, text=self.tr("model3d_btn_run_python"), command=self.run_python_command, width=140).grid(row=0, column=2, padx=10, pady=8)

        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            self.open_output_button.configure(state="normal")
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, str(self.output_base))

    def label_for_key(self, key):
        for backend_key, label in self.backends:
            if backend_key == key:
                return label
        return self.backends[0][1]

    def key_for_label(self, label):
        for backend_key, backend_label in self.backends:
            if backend_label == label:
                return backend_key
        return self.backends[0][0]

    def on_backend_change(self, selected_label):
        self.backend_key = self.key_for_label(selected_label)
        self.update_backend_ui()
        self.update_weights_menu()

    def update_backend_ui(self):
        if self.backend_key in ("stepx1", "hunyuan3d2", "sam3d"):
            self.input_mode_var.set(self.input_mode_labels["single"])
            self.input_mode_menu.configure(state="disabled")
        else:
            self.input_mode_menu.configure(state="normal")

        supports_weights = self.backend_key in ("stepx1", "hunyuan3d2", "sam3d")
        weights_state = "normal" if supports_weights else "disabled"
        self.install_weights_button.configure(state=weights_state)
        self.uninstall_weights_button.configure(state=weights_state)
        self.weights_menu.configure(state=weights_state)
        self.update_action_buttons()

    def get_input_mode(self):
        label = self.input_mode_var.get()
        for key, value in self.input_mode_labels.items():
            if value == label:
                return key
        return "single"

    def select_inputs(self):
        mode = self.get_input_mode()
        paths = []
        if self.backend_key == "sam3d":
            image_path = filedialog.askopenfilename(
                title=self.tr("model3d_input_placeholder"),
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
            )
            if not image_path:
                self.input_paths = []
                self.render_selected_inputs()
                return
            mask_path = filedialog.askopenfilename(
                title=self.tr("model3d_mask_placeholder"),
                filetypes=[("Masks", "*.png *.jpg *.jpeg *.bmp")],
            )
            if not mask_path:
                self.input_paths = []
                self.render_selected_inputs()
                return
            paths = [image_path, mask_path]
        elif mode == "multi":
            paths = list(
                filedialog.askopenfilenames(
                    title=self.tr("model3d_input_folder_placeholder"),
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
                )
            )
        elif mode == "video":
            path = filedialog.askopenfilename(
                title=self.tr("model3d_input_video_placeholder"),
                filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv")],
            )
            if path:
                paths = [path]
        else:
            path = filedialog.askopenfilename(
                title=self.tr("model3d_input_placeholder"),
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
            )
            if path:
                paths = [path]

        self.input_paths = paths
        self.render_selected_inputs()

    def browse_output(self):
        path = filedialog.askdirectory(title=self.tr("model3d_output_label"))
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def open_backend_link(self):
        url = self.backend_links.get(self.backend_key)
        if self.backend_key == "reconv":
            url = self.backend_spaces.get("reconv", url)
        if url:
            webbrowser.open(url)

    def open_hf_access(self):
        option = self.get_selected_weight_option()
        repo_id = option["repo_id"] if option else ""
        if self.backend_key == "sam3d":
            repo_id = "facebook/sam-3d-objects"
        if repo_id:
            webbrowser.open(f"https://huggingface.co/{repo_id}")
        messagebox.showinfo(self.tr("model3d_hf_title"), self.tr("model3d_hf_instructions"))

    def open_hf_tokens(self):
        webbrowser.open("https://huggingface.co/settings/tokens")

    def open_output_folder(self):
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            os.startfile(self.last_output_dir)

    def open_generados_folder(self):
        if self.output_base.exists():
            os.startfile(str(self.output_base))

    def render_selected_inputs(self):
        self.selected_box.configure(state="normal")
        self.selected_box.delete("0.0", "end")
        if self.input_paths:
            if self.backend_key == "sam3d" and len(self.input_paths) >= 2:
                self.selected_box.insert("end", f"{self.tr('model3d_label_image')}: {self.input_paths[0]}\n")
                self.selected_box.insert("end", f"{self.tr('model3d_label_mask')}: {self.input_paths[1]}")
            else:
                self.selected_box.insert("end", "\n".join(self.input_paths))
        self.selected_box.configure(state="disabled")

    def run_python_command(self):
        cmd = self.py_cmd_entry.get().strip()
        if not cmd:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_missing_py_cmd"))
            return
        thread = threading.Thread(target=self._run_python_command_worker, args=(cmd,), daemon=True)
        thread.start()

    def _run_python_command_worker(self, cmd):
        python_path = sys.executable.replace("pythonw.exe", "python.exe")
        self.safe_log(self.tr("model3d_msg_python_running"))
        self.execute_python([python_path, "-c", cmd], cwd=str(self.app_root))

    def update_weights_menu(self):
        self.weight_options = self.get_weights_options(self.backend_key)
        labels = [opt["label"] for opt in self.weight_options]
        if labels:
            self.weight_var.set(labels[0])
        else:
            self.weight_var.set(self.tr("model3d_weights_none"))
            labels = [self.tr("model3d_weights_none")]
        self.weights_menu.configure(values=labels)
        self.update_action_buttons()

    def on_weight_change(self, _value):
        self.update_action_buttons()

    def get_selected_weight_option(self):
        selected = self.weight_var.get()
        for opt in self.weight_options:
            if opt["label"] == selected:
                return opt
        return self.weight_options[0] if self.weight_options else None

    def start_generate(self):
        self.config["backend"] = self.backend_key
        self.save_config()

        mode = self.get_input_mode()
        if not self.input_paths:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_missing_input"))
            return
        if any(not os.path.exists(p) for p in self.input_paths):
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_missing_input"))
            return
        if self.backend_key == "stepx1" and mode != "single":
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_stepx1_single"))
            return
        if self.backend_key == "sam3d" and len(self.input_paths) < 2:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_sam3d_needs_mask"))
            return
        if self.backend_key == "hunyuan3d2" and mode != "single":
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_hunyuan_single"))
            return

        output_dir = self.output_entry.get().strip()
        if not output_dir:
            output_dir = str(self.output_base)
        if Path(output_dir).resolve() == self.output_base.resolve():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = str(self.output_base / f"model3d_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        python_path = sys.executable.replace("pythonw.exe", "python.exe")

        self.generate_button.configure(state="disabled")
        if self.backend_key == "stepx1":
            repo_path = self.resolve_repo_path("stepx1")
            if not repo_path:
                self.warn_missing_repo("Step1X-3D", "https://github.com/stepfun-ai/Step1X-3D")
                self.generate_button.configure(state="normal")
                return
            thread = threading.Thread(
                target=self.run_stepx1,
                args=(self.input_paths[0], output_dir, repo_path, python_path),
                daemon=True,
            )
            thread.start()
            return

        if self.backend_key == "hunyuan3d2":
            repo_path = self.resolve_repo_path("hunyuan3d2")
            if not repo_path:
                self.warn_missing_repo("Hunyuan3D-2", "https://github.com/Tencent/Hunyuan3D-2")
                self.generate_button.configure(state="normal")
                return
            thread = threading.Thread(
                target=self.run_hunyuan,
                args=(self.input_paths[0], output_dir, repo_path, python_path),
                daemon=True,
            )
            thread.start()
            return

        if self.backend_key == "sam3d":
            repo_path = self.resolve_repo_path("sam3d")
            if not repo_path:
                self.warn_missing_repo("sam-3d-objects", "https://github.com/facebookresearch/sam-3d-objects")
                self.generate_button.configure(state="normal")
                return
            thread = threading.Thread(
                target=self.run_sam3d,
                args=(self.input_paths[0], self.input_paths[1], output_dir, repo_path, python_path),
                daemon=True,
            )
            thread.start()
            return

        self.log(self.tr("model3d_msg_running"))
        space_url = self.backend_spaces.get(self.backend_key)
        if space_url:
            webbrowser.open(space_url)
        self.last_output_dir = output_dir
        self.config["last_output_dir"] = output_dir
        self.save_config()
        self.open_output_button.configure(state="normal")
        self.generate_button.configure(state="normal")

    def run_stepx1(self, input_path, output_dir, repo_path, python_path):
        try:
            self.safe_log(self.tr("model3d_msg_running"))
            script_path = self.write_stepx1_script(input_path, output_dir, repo_path)
            ok = self.run_script_with_retry(script_path, repo_path, python_path)
            if ok:
                self.safe_log(self.tr("model3d_msg_finished"))
                self.last_output_dir = output_dir
                self.config["last_output_dir"] = output_dir
                self.save_config()
                self.after(0, lambda: self.open_output_button.configure(state="normal"))
            else:
                self.safe_log(self.tr("model3d_msg_failed").format(1))
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")
        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def run_hunyuan(self, input_path, output_dir, repo_path, python_path):
        try:
            self.safe_log(self.tr("model3d_msg_running"))
            script_path = self.write_hunyuan_script(input_path, output_dir, repo_path)
            ok = self.run_script_with_retry(script_path, repo_path, python_path)
            if ok:
                self.safe_log(self.tr("model3d_msg_finished"))
                self.last_output_dir = output_dir
                self.config["last_output_dir"] = output_dir
                self.save_config()
                self.after(0, lambda: self.open_output_button.configure(state="normal"))
            else:
                self.safe_log(self.tr("model3d_msg_failed").format(1))
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")
        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def run_sam3d(self, image_path, mask_path, output_dir, repo_path, python_path):
        try:
            self.safe_log(self.tr("model3d_msg_running"))
            script_path = self.write_sam3d_script(image_path, mask_path, output_dir, repo_path)
            ok = self.run_script_with_retry(script_path, repo_path, python_path)
            if ok:
                self.safe_log(self.tr("model3d_msg_finished"))
                self.last_output_dir = output_dir
                self.config["last_output_dir"] = output_dir
                self.save_config()
                self.after(0, lambda: self.open_output_button.configure(state="normal"))
            else:
                self.safe_log(self.tr("model3d_msg_failed").format(1))
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")
        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def write_stepx1_script(self, input_path, output_dir, repo_path):
        script_path = self.data_dir / "stepx1_run.py"
        script = (
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, r\"{repo}\")\n"
            "import torch\n"
            "import trimesh\n"
            "from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline\n"
            "from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face\n"
            "from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline\n"
            "input_image = r\"{input}\"\n"
            "out_dir = r\"{output}\"\n"
            "os.makedirs(out_dir, exist_ok=True)\n"
            "weights_dir = r\"{weights}\"\n"
            "base = weights_dir if weights_dir and os.path.exists(weights_dir) else \"stepfun-ai/Step1X-3D\"\n"
            "geo = Step1X3DGeometryPipeline.from_pretrained(\n"
            "    base, subfolder=\"Step1X-3D-Geometry-1300m\"\n"
            ").to(\"cuda\")\n"
            "gen = torch.Generator(device=geo.device).manual_seed(2025)\n"
            "out = geo(input_image, guidance_scale=7.5, num_inference_steps=50, generator=gen)\n"
            "mesh_path = os.path.join(out_dir, \"mesh.glb\")\n"
            "out.mesh[0].export(mesh_path)\n"
            "tex = Step1X3DTexturePipeline.from_pretrained(\n"
            "    base, subfolder=\"Step1X-3D-Texture\"\n"
            ")\n"
            "mesh = trimesh.load(mesh_path)\n"
            "mesh = remove_degenerate_face(mesh)\n"
            "mesh = reduce_face(mesh)\n"
            "textured = tex(input_image, mesh, seed=2025)\n"
            "textured.export(os.path.join(out_dir, \"mesh_textured.glb\"))\n"
        ).format(repo=repo_path, input=input_path, output=output_dir, weights=self.get_weights_dir("stepx1"))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        return str(script_path)

    def write_hunyuan_script(self, input_path, output_dir, repo_path):
        script_path = self.data_dir / "hunyuan_run.py"
        script = (
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, r\"{repo}\")\n"
            "os.environ[\"HF_HUB_DISABLE_PROGRESS_BARS\"] = \"1\"\n"
            "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline\n"
            "from hy3dgen.texgen import Hunyuan3DPaintPipeline\n"
            "input_image = r\"{input}\"\n"
            "out_dir = r\"{output}\"\n"
            "os.makedirs(out_dir, exist_ok=True)\n"
            "mesh_path = os.path.join(out_dir, \"mesh.glb\")\n"
            "weights_dir = r\"{weights}\"\n"
            "base = \"tencent/Hunyuan3D-2\"\n"
            "local_ok = False\n"
            "if weights_dir and os.path.exists(weights_dir):\n"
            "    os.environ[\"HY3DGEN_MODELS\"] = weights_dir\n"
            "    for name in (\n"
            "        \"hunyuan3d-dit-v2-0\",\n"
            "        \"hunyuan3d-dit-v2-0-fast\",\n"
            "        \"hunyuan3d-dit-v2-0-turbo\",\n"
            "    ):\n"
            "        path = os.path.join(weights_dir, \"tencent\", \"Hunyuan3D-2\", name)\n"
            "        if os.path.exists(path):\n"
            "            local_ok = True\n"
            "            break\n"
            "if local_ok:\n"
            "    os.environ[\"HF_HUB_OFFLINE\"] = \"1\"\n"
            "pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(base)\n"
            "mesh = pipeline(image=input_image)[0]\n"
            "mesh.export(mesh_path)\n"
            "paint = Hunyuan3DPaintPipeline.from_pretrained(base)\n"
            "mesh = paint(mesh, image=input_image)\n"
            "mesh.export(os.path.join(out_dir, \"mesh_textured.glb\"))\n"
        ).format(repo=repo_path, input=input_path, output=output_dir, weights=self.get_weights_dir("hunyuan3d2"))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        return str(script_path)

    def write_sam3d_script(self, image_path, mask_path, output_dir, repo_path):
        script_path = self.data_dir / "sam3d_run.py"
        script = (
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, r\"{repo}\")\n"
            "from notebook.inference import Inference, load_image, load_mask\n"
            "image_path = r\"{image}\"\n"
            "mask_path = r\"{mask}\"\n"
            "out_dir = r\"{output}\"\n"
            "os.makedirs(out_dir, exist_ok=True)\n"
            "config_path = os.path.join(\"checkpoints\", \"hf\", \"pipeline.yaml\")\n"
            "inference = Inference(config_path, compile=False)\n"
            "image = load_image(image_path)\n"
            "mask = load_mask(mask_path)\n"
            "output = inference(image, mask, seed=42)\n"
            "output[\"gs\"].save_ply(os.path.join(out_dir, \"sam3d_splat.ply\"))\n"
        ).format(repo=repo_path, image=image_path, mask=mask_path, output=output_dir)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        return str(script_path)

    def install_backend_action(self):
        self.run_manage_thread(self.install_backend, self.tr("model3d_status_install_backend"))

    def uninstall_backend_action(self):
        self.run_manage_thread(self.uninstall_backend, self.tr("model3d_status_uninstall_backend"))

    def install_weights_action(self):
        label = self.get_selected_weight_option()
        detail = self.tr("model3d_status_install_weights")
        if label:
            detail = detail.format(label["label"])
        self.run_manage_thread(self.install_weights, detail)

    def uninstall_weights_action(self):
        label = self.get_selected_weight_option()
        detail = self.tr("model3d_status_uninstall_weights")
        if label:
            detail = detail.format(label["label"])
        self.run_manage_thread(self.uninstall_weights, detail)

    def run_manage_thread(self, func, status_text):
        for btn in (self.install_backend_button, self.uninstall_backend_button, self.install_weights_button, self.uninstall_weights_button):
            btn.configure(state="disabled")
        self.set_task_status(True, status_text)
        thread = threading.Thread(target=self.run_manage_task, args=(func,), daemon=True)
        thread.start()

    def run_manage_task(self, func):
        try:
            if self.backend_key == "reconv":
                self.safe_log(self.tr("model3d_msg_backend_not_supported"))
                return
            func()
        finally:
            self.after(0, lambda: self.restore_manage_buttons())

    def restore_manage_buttons(self):
        self.set_task_status(False)
        self.update_action_buttons()

    def install_backend(self):
        repo_path = self.get_repo_path(self.backend_key)
        if not repo_path:
            self.safe_log(self.tr("model3d_msg_backend_not_supported"))
            return
        if os.path.exists(repo_path):
            self.safe_log(self.tr("model3d_msg_backend_exists"))
            return
        repo_url = self.get_repo_url(self.backend_key)
        if not repo_url:
            self.safe_log(self.tr("model3d_msg_backend_not_supported"))
            return
        base_dir = os.path.dirname(repo_path)
        os.makedirs(base_dir, exist_ok=True)
        if not self.check_git_available():
            self.safe_log(self.tr("model3d_msg_git_missing"))
            return
        self.safe_log(self.tr("model3d_msg_backend_installing").format(repo_url))
        self.run_process(["git", "clone", "--depth", "1", repo_url, repo_path])
        self.after(0, self.update_action_buttons)

    def uninstall_backend(self):
        repo_path = self.get_repo_path(self.backend_key)
        if not repo_path or not os.path.exists(repo_path):
            self.safe_log(self.tr("model3d_msg_backend_missing"))
            return
        try:
            import shutil
            shutil.rmtree(repo_path)
            self.safe_log(self.tr("model3d_msg_backend_removed"))
            self.after(0, self.update_action_buttons)
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")

    def install_weights(self):
        if self.backend_key == "sam3d":
            repo_path = self.get_repo_path("sam3d")
            if not repo_path or not os.path.exists(repo_path):
                self.safe_log(self.tr("model3d_msg_backend_missing"))
                return
            self.install_sam3d_weights(repo_path)
            self.after(0, self.update_action_buttons)
            return

        option = self.get_selected_weight_option()
        if not option:
            self.safe_log(self.tr("model3d_msg_weights_not_supported"))
            return
        weights_dir = option["local_dir"]
        repo_id = option["repo_id"]
        if not weights_dir or not repo_id:
            self.safe_log(self.tr("model3d_msg_weights_not_supported"))
            return
        if os.path.exists(weights_dir):
            self.safe_log(self.tr("model3d_msg_weights_exists"))
            return
        self.safe_log(self.tr("model3d_msg_weights_installing").format(repo_id))
        self.download_hf_repo(repo_id, weights_dir)
        self.after(0, self.update_action_buttons)

    def uninstall_weights(self):
        if self.backend_key == "sam3d":
            repo_path = self.get_repo_path("sam3d")
            if not repo_path or not os.path.exists(repo_path):
                self.safe_log(self.tr("model3d_msg_backend_missing"))
                return
            checkpoints = os.path.join(repo_path, "checkpoints", "hf")
            if not os.path.exists(checkpoints):
                self.safe_log(self.tr("model3d_msg_weights_missing"))
                return
            try:
                import shutil
                shutil.rmtree(checkpoints)
                self.safe_log(self.tr("model3d_msg_weights_removed"))
                self.after(0, self.update_action_buttons)
            except Exception as exc:
                self.safe_log(f"{self.tr('status_error')}: {exc}")
            return

        option = self.get_selected_weight_option()
        weights_dir = option["local_dir"] if option else ""
        if not weights_dir or not os.path.exists(weights_dir):
            self.safe_log(self.tr("model3d_msg_weights_missing"))
            return
        try:
            import shutil
            shutil.rmtree(weights_dir)
            self.safe_log(self.tr("model3d_msg_weights_removed"))
            self.after(0, self.update_action_buttons)
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")

    def install_sam3d_weights(self, repo_path):
        checkpoints_dir = os.path.join(repo_path, "checkpoints", "hf")
        if os.path.exists(checkpoints_dir):
            self.safe_log(self.tr("model3d_msg_weights_exists"))
            return
        tmp_dir = os.path.join(repo_path, "checkpoints", "hf-download")
        self.safe_log(self.tr("model3d_msg_weights_installing").format("facebook/sam-3d-objects"))
        ok = self.download_hf_repo("facebook/sam-3d-objects", tmp_dir)
        if not ok:
            return
        try:
            import shutil
            src = os.path.join(tmp_dir, "checkpoints")
            if not os.path.exists(src):
                self.safe_log(self.tr("model3d_msg_weights_missing"))
                return
            os.makedirs(os.path.dirname(checkpoints_dir), exist_ok=True)
            shutil.move(src, checkpoints_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self.safe_log(self.tr("model3d_msg_weights_installed"))
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")

    def download_hf_repo(self, repo_id, local_dir):
        try:
            from huggingface_hub import HfApi, HfFolder, hf_hub_download
        except Exception:
            self.safe_log(self.tr("model3d_msg_hf_missing"))
            return False
        try:
            os.makedirs(local_dir, exist_ok=True)
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            token = HfFolder.get_token()
            if not token:
                self.safe_log(self.tr("model3d_hf_missing_token"))
                return False
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            total = max(len(files), 1)
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            temp_stdout = None
            temp_stderr = None
            try:
                if sys.stdout is None:
                    temp_stdout = open(os.devnull, "w")
                    sys.stdout = temp_stdout
                if sys.stderr is None:
                    temp_stderr = open(os.devnull, "w")
                    sys.stderr = temp_stderr
                for idx, filename in enumerate(files, start=1):
                    percent = int((idx / total) * 100)
                    self.update_task_detail(self.tr("model3d_status_download_weights").format(percent))
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="model",
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        token=token,
                    )
            finally:
                if temp_stdout:
                    temp_stdout.close()
                if temp_stderr:
                    temp_stderr.close()
                sys.stdout, sys.stderr = orig_stdout, orig_stderr
            self.safe_log(self.tr("model3d_msg_weights_installed"))
            return True
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")
            if "401" in str(exc):
                self.after(0, lambda: messagebox.showwarning(self.tr("model3d_hf_title"), self.tr("model3d_hf_gated")))
            return False

    def run_script_with_retry(self, script_path, repo_path, python_path):
        max_attempts = 6
        attempt = 0
        while attempt < max_attempts:
            code, missing = self.execute_python([python_path, script_path], cwd=repo_path)
            if code == 0:
                return True
            if not missing:
                return False
            self.safe_log(self.tr("model3d_msg_auto_install").format(", ".join(missing)))
            installed = self.install_python_packages(missing, python_path)
            if not installed:
                return False
            attempt += 1
            self.safe_log(self.tr("model3d_msg_retrying"))
        return False

    def execute_python(self, cmd, cwd):
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        missing = []
        for line in process.stdout:
            line = line.rstrip()
            self.safe_log(line)
            match = re.search(r"No module named '([^']+)'", line)
            if match:
                missing.append(match.group(1))
                continue
            match = re.search(r"([a-zA-Z0-9_\-]+)>=([0-9\.]+) is required", line)
            if match:
                missing.append(f"{match.group(1)}>={match.group(2)}")
        process.wait()
        missing = list(dict.fromkeys(missing))
        return process.returncode, missing

    def install_python_packages(self, packages, python_path):
        if not packages:
            return False
        mapped = []
        for pkg in packages:
            if pkg == "skimage":
                mapped.append("scikit-image")
            else:
                mapped.append(pkg)
        mapped = list(dict.fromkeys(mapped))
        self.safe_log(self.tr("model3d_msg_installing_deps").format(", ".join(mapped)))
        cmd = [python_path, "-m", "pip", "install", "--upgrade"] + mapped
        code, _ = self.execute_python(cmd, cwd=str(self.app_root))
        return code == 0

    def set_task_status(self, running, detail_text=None):
        self._task_running = running
        if running:
            text = self.tr("model3d_status_working")
            if detail_text:
                text = self.tr("model3d_status_working_detail").format(detail_text)
            self.status_label.configure(text=text, text_color="orange")
        else:
            self.status_label.configure(text=self.tr("model3d_status_idle"), text_color="gray")

    def update_task_detail(self, detail_text):
        if not self._task_running:
            return
        text = self.tr("model3d_status_working_detail").format(detail_text)
        self.after(0, lambda: self.status_label.configure(text=text, text_color="orange"))

    def save_hf_token(self):
        token = self.hf_token_var.get().strip()
        if not token:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_hf_missing_token"))
            return
        try:
            from huggingface_hub import HfFolder
        except Exception:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_hf_missing"))
            return
        try:
            HfFolder.save_token(token)
            messagebox.showinfo(self.tr("model3d_hf_title"), self.tr("model3d_hf_saved"))
            self.refresh_hf_token_ui()
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{exc}")

    def delete_hf_token(self):
        try:
            from huggingface_hub import HfFolder
        except Exception:
            messagebox.showwarning(self.tr("status_error"), self.tr("model3d_msg_hf_missing"))
            return
        try:
            HfFolder.delete_token()
            self.hf_token_var.set("")
            messagebox.showinfo(self.tr("model3d_hf_title"), self.tr("model3d_hf_deleted"))
            self.refresh_hf_token_ui()
        except Exception as exc:
            messagebox.showwarning(self.tr("status_error"), f"{exc}")

    def refresh_hf_token_ui(self):
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
        if token:
            self.hf_token_entry.grid_remove()
            self.hf_token_label.configure(text=self.tr("model3d_hf_token_saved"))
            self.hf_token_button.configure(text=self.tr("model3d_btn_delete_hf"), command=self.delete_hf_token)
        else:
            self.hf_token_entry.grid()
            self.hf_token_label.configure(text=self.tr("model3d_hf_token_label"))
            self.hf_token_button.configure(text=self.tr("model3d_btn_save_hf"), command=self.save_hf_token)

    def update_action_buttons(self):
        if self.backend_key == "reconv":
            self.install_backend_button.pack_forget()
            self.uninstall_backend_button.pack_forget()
            self.install_weights_button.pack_forget()
            self.uninstall_weights_button.pack_forget()
            self.weights_menu.configure(state="disabled")
            return

        backend_installed = self.is_backend_installed(self.backend_key)
        self.toggle_button(self.install_backend_button, not backend_installed)
        self.toggle_button(self.uninstall_backend_button, backend_installed)

        supports_weights = self.backend_key in ("stepx1", "hunyuan3d2", "sam3d")
        if supports_weights:
            weights_installed = self.is_weights_installed(self.backend_key)
            self.toggle_button(self.install_weights_button, not weights_installed)
            self.toggle_button(self.uninstall_weights_button, weights_installed)
        else:
            self.install_weights_button.pack_forget()
            self.uninstall_weights_button.pack_forget()
            self.weights_menu.configure(state="disabled")

    def toggle_button(self, button, show):
        if show:
            if not button.winfo_manager():
                button.pack(side="left", padx=5)
        else:
            if button.winfo_manager():
                button.pack_forget()

    def is_backend_installed(self, backend_key):
        repo_path = self.get_repo_path(backend_key)
        return bool(repo_path and os.path.exists(repo_path))

    def is_weights_installed(self, backend_key):
        if backend_key == "sam3d":
            repo_path = self.get_repo_path("sam3d")
            if not repo_path:
                return False
            return os.path.exists(os.path.join(repo_path, "checkpoints", "hf"))
        option = self.get_selected_weight_option()
        if not option:
            return False
        local_dir = option.get("local_dir") or ""
        return bool(local_dir and os.path.exists(local_dir))

    def get_repo_path(self, backend_key):
        repo_name = {
            "stepx1": "Step1X-3D",
            "hunyuan3d2": "Hunyuan3D-2",
            "sam3d": "sam-3d-objects",
        }.get(backend_key)
        if not repo_name:
            return None
        return str(self.app_root / "system" / "3d-backends" / repo_name)

    def get_repo_url(self, backend_key):
        return {
            "stepx1": "https://github.com/stepfun-ai/Step1X-3D",
            "hunyuan3d2": "https://github.com/Tencent/Hunyuan3D-2",
            "sam3d": "https://github.com/facebookresearch/sam-3d-objects",
        }.get(backend_key)

    def get_weights_options(self, backend_key):
        suffix = self.tr("suffix_recommended")
        base_dir = self.app_root / "system" / "3d-weights"
        if backend_key == "stepx1":
            return [{
                "key": "default",
                "label": f"stepfun-ai/Step1X-3D{suffix}",
                "repo_id": "stepfun-ai/Step1X-3D",
                "local_dir": str(base_dir / "Step1X-3D"),
            }]
        if backend_key == "hunyuan3d2":
            return [{
                "key": "default",
                "label": f"tencent/Hunyuan3D-2{suffix}",
                "repo_id": "tencent/Hunyuan3D-2",
                "local_dir": str(base_dir / "Hunyuan3D-2"),
            }]
        if backend_key == "sam3d":
            return [{
                "key": "default",
                "label": f"facebook/sam-3d-objects{suffix}",
                "repo_id": "facebook/sam-3d-objects",
                "local_dir": "",
            }]
        return []

    def get_weights_dir(self, backend_key):
        name = {
            "stepx1": "Step1X-3D",
            "hunyuan3d2": "Hunyuan3D-2",
        }.get(backend_key)
        if not name:
            return ""
        return str(self.app_root / "system" / "3d-weights" / name)

    def resolve_repo_path(self, backend_key):
        repo_name = {
            "stepx1": "Step1X-3D",
            "hunyuan3d2": "Hunyuan3D-2",
            "sam3d": "sam-3d-objects",
        }.get(backend_key)
        if not repo_name:
            return None

        candidates = [
            self.app_root / "system" / "3d-backends" / repo_name,
            self.app_root / "system" / repo_name,
            self.app_root / repo_name,
            self.app_root / "backends" / repo_name,
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def warn_missing_repo(self, name, url):
        expected = self.app_root / "system" / "3d-backends" / name
        message = self.tr("model3d_msg_missing_repo").format(name, expected, url)
        messagebox.showwarning(self.tr("status_error"), message)

    def check_git_available(self):
        try:
            result = subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def run_process(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            for line in process.stdout:
                self.safe_log(line.rstrip())
            process.wait()
            if process.returncode == 0:
                self.safe_log(self.tr("model3d_msg_backend_installed"))
            else:
                self.safe_log(self.tr("model3d_msg_backend_failed").format(process.returncode))
        except Exception as exc:
            self.safe_log(f"{self.tr('status_error')}: {exc}")

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))

    def load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass