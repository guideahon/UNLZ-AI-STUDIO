"""
Minimal Tkinter UI to monitor presets, logs and tester feedback.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional


def _safe_import_tk():  # pragma: no cover - UI helper
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk, filedialog
        from tkinter.scrolledtext import ScrolledText
        return tk, ttk, messagebox, filedialog, ScrolledText
    except Exception as exc:
        raise RuntimeError(f"Tkinter no disponible: {exc}")


class StudioGUI:
    def __init__(
        self,
        profile_manager,
        preflight_report: Dict,
        log_dir: Path,
        logo_path: Optional[Path] = None,
        manager=None,
        warm_callback=None,
        splash_path: Optional[Path] = None,
    ):
        self.profile_manager = profile_manager
        self.preflight_report = preflight_report
        self.log_dir = log_dir
        self.tk, self.ttk, self.messagebox, self.filedialog, self.ScrolledText = _safe_import_tk()
        self.manager = manager
        self.warm_callback = warm_callback
        self._splash_image = None
        self._splash_window = None

        self.root = self.tk.Tk()
        self.root.title("UNLZ AI Studio - Monitor")
        self.root.geometry("1024x720")
        self.root.minsize(960, 680)
        if logo_path and logo_path.exists():
            try:
                self.root.iconbitmap(default=str(logo_path))
            except Exception:
                pass
        self._init_style()
        if splash_path and splash_path.exists():
            self._show_splash(splash_path)
        else:
            self.root.deiconify()
        self.status_var = self.tk.StringVar(value="Listo.")
        self.profile_var = self.tk.StringVar(value=self.profile_manager.active_profile.key)
        self.force_var = self.tk.BooleanVar(value=False)
        self.feedback_issue_var = self.tk.StringVar(value="ok")
        self.custom_model_var = self.tk.StringVar()
        self.custom_ctx_var = self.tk.StringVar()
        self.custom_ngl_var = self.tk.StringVar()
        self.custom_threads_var = self.tk.StringVar()
        self.custom_batch_var = self.tk.StringVar()
        self.custom_tensor_var = self.tk.StringVar()
        self.custom_rope_base_var = self.tk.StringVar()
        self.custom_rope_scale_var = self.tk.StringVar()
        self.custom_temp_var = self.tk.StringVar()
        self.custom_top_p_var = self.tk.StringVar()
        self.custom_top_k_var = self.tk.StringVar()
        self.custom_repeat_var = self.tk.StringVar()
        self.custom_flash_var = self.tk.BooleanVar()
        self.server_status_var = self.tk.StringVar(value="Servidor LLM: detenido")
        self._server_busy = False
        self._server_thread: Optional[threading.Thread] = None
        self.endpoint_vars: Dict[str, object] = {}
        self.endpoint_settings_vars: Dict[str, Dict[str, object]] = {}
        self._endpoint_updating = False

        self._build_layout()
        self._refresh_profile_summary()
        self._refresh_feedback_summary()
        self._refresh_suggestions()
        self._custom_visible = False
        self._populate_custom_fields()
        self._show_custom_frame(self.profile_var.get() == "personalizado")
        self._sync_endpoint_checkboxes()
        self._populate_endpoint_settings()
        self._refresh_server_status()
        self._schedule_updates()

    # --- Layout -----------------------------------------------------
    def _init_style(self):
        self.style = self.ttk.Style(self.root)
        preferred = ("vista", "winnative", "xpnative", "clam")
        for theme in preferred:
            if theme in self.style.theme_names():
                self.style.theme_use(theme)
                break
        font = ("Segoe UI", 10)
        bg = "#f2f3f7"
        fg = "#1f2328"
        accent = "#3a6ff7"
        green = "#2ecc71"
        red = "#e65054"
        self.root.option_add("*Font", font)
        self.root.configure(background=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabelframe", background=bg, foreground=fg, relief="flat")
        self.style.configure("TLabelframe.Label", background=bg, foreground=fg, font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background=bg, foreground=fg, font=font)
        self.style.configure("TButton", padding=8, font=("Segoe UI Semibold", 10))
        self.style.map("TButton", background=[("active", accent)], foreground=[("active", "#ffffff")])
        self.style.configure("TNotebook", background=bg)
        self.style.configure("TNotebook.Tab", padding=(12, 6, 12, 6), font=font)
        self.style.configure("TCombobox", padding=6)
        self.style.configure("TProgressbar", background=accent, troughcolor="#dce1ed", bordercolor=bg)
        self.style.configure("Running.Horizontal.TProgressbar", background=green, troughcolor="#dce1ed", bordercolor=bg)
        self.style.configure("Stopped.Horizontal.TProgressbar", background=red, troughcolor="#f6d4d6", bordercolor=bg)
        self.style.configure("Busy.Horizontal.TProgressbar", background=accent, troughcolor="#dce1ed", bordercolor=bg)

    def _show_splash(self, splash_path: Path):
        try:
            self.root.withdraw()
            splash = self.tk.Toplevel(self.root)
            splash.overrideredirect(True)
            img = self.tk.PhotoImage(file=str(splash_path))
            self._splash_image = img
            label = self.tk.Label(splash, image=img, borderwidth=0, highlightthickness=0, background="#ffffff")
            label.pack()
            splash.update_idletasks()
            w = img.width()
            h = img.height()
            sw = splash.winfo_screenwidth()
            sh = splash.winfo_screenheight()
            x = (sw - w) // 2
            y = (sh - h) // 2
            splash.geometry(f"{w}x{h}+{x}+{y}")
            self._splash_window = splash
            self.root.after(2200, self._close_splash)
        except Exception:
            self.root.deiconify()

    def _close_splash(self):
        if self._splash_window is not None:
            try:
                self._splash_window.destroy()
            except Exception:
                pass
            self._splash_window = None
        self.root.deiconify()

    def _build_layout(self):
        top = self.ttk.Frame(self.root, padding=10)
        top.pack(side=self.tk.TOP, fill=self.tk.X)

        self._build_system_info(top)
        self._build_profile_controls()
        self._build_notebook()

        status_bar = self.ttk.Label(self.root, textvariable=self.status_var, relief=self.tk.SUNKEN, anchor="w")
        status_bar.pack(side=self.tk.BOTTOM, fill=self.tk.X)

    def _build_system_info(self, container):
        info_frame = self.ttk.LabelFrame(container, text="Hardware detectado", padding=10)
        info_frame.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)
        system_info = self.profile_manager.system_info.to_display_dict()
        row = 0
        for key, value in system_info.items():
            self.ttk.Label(info_frame, text=f"{key}:").grid(row=row, column=0, sticky="w", padx=4, pady=2)
            self.ttk.Label(info_frame, text=value).grid(row=row, column=1, sticky="w", padx=4, pady=2)
            row += 1

    def _build_profile_controls(self):
        frame = self.ttk.LabelFrame(self.root, text="Perfiles y presets", padding=10)
        frame.pack(side=self.tk.TOP, fill=self.tk.X, padx=10, pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=0)

        preset_options = []
        self._preset_key_by_label: Dict[str, str] = {}
        for preset in self.profile_manager.list_presets():
            label = f"{preset['title']} ({preset['key']}) - {preset['status']}"
            preset_options.append(label)
            self._preset_key_by_label[label] = preset["key"]

        self.ttk.Label(frame, text="Seleccionar perfil:").grid(row=0, column=0, sticky="w")
        self.preset_box = self.ttk.Combobox(frame, values=preset_options, state="readonly", width=60)
        selected_label = next(
            (label for label, key in self._preset_key_by_label.items() if key == self.profile_var.get()),
            preset_options[0] if preset_options else "",
        )
        if selected_label:
            self.preset_box.set(selected_label)
        self.preset_box.grid(row=0, column=1, sticky="we", padx=6)
        self.preset_box.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self.force_check = self.ttk.Checkbutton(frame, text="Forzar si no cumple requisitos", variable=self.force_var)
        self.force_check.grid(row=0, column=2, padx=6)

        apply_btn = self.ttk.Button(frame, text="Aplicar", command=self._on_apply_profile)
        apply_btn.grid(row=0, column=3, padx=6)

        refresh_btn = self.ttk.Button(frame, text="Actualizar listado", command=self._refresh_presets)
        refresh_btn.grid(row=0, column=4, padx=6)

        summary_frame = self.ttk.LabelFrame(frame, text="Resumen del perfil activo", padding=6)
        summary_frame.grid(row=1, column=0, columnspan=5, sticky="we", pady=(8, 0))

        self.profile_summary = self.ScrolledText(summary_frame, height=4, wrap="word")
        self.profile_summary.pack(fill=self.tk.X)
        self.profile_summary.configure(font=("Segoe UI", 10), background="#ffffff", relief="flat", borderwidth=0)

        endpoint_cfg = self.profile_manager.get_endpoint_config()
        endpoint_frame = self.ttk.LabelFrame(frame, text="Endpoints activos", padding=6)
        endpoint_frame.grid(row=2, column=0, columnspan=5, sticky="we", pady=(6, 0))
        endpoint_labels = {
            "llm": "LLM (llama.cpp)",
            "clm": "CLM (HF en-proc)",
            "vlm": "VLM (LMDeploy/HF)",
            "alm": "ALM (Audio pipeline)",
            "slm": "SLM (Audio streaming)",
        }
        for idx, name in enumerate(["llm", "clm", "vlm", "alm", "slm"]):
            var = self.tk.BooleanVar(value=endpoint_cfg.get(name, False))
            chk = self.ttk.Checkbutton(
                endpoint_frame,
                text=endpoint_labels.get(name, name.upper()),
                variable=var,
                command=lambda n=name: self._on_endpoint_toggle(n),
            )
            chk.grid(row=0, column=idx, padx=6, pady=2, sticky="w")
            self.endpoint_vars[name] = var
        for col in range(5):
            endpoint_frame.columnconfigure(col, weight=1)

        server_frame = self.ttk.Frame(frame)
        server_frame.grid(row=3, column=0, columnspan=5, sticky="we", pady=(6, 0))
        self.server_button = self.ttk.Button(server_frame, text="Iniciar servidor", command=self._on_server_toggle)
        self.server_button.pack(side=self.tk.LEFT)
        self.server_progress = self.ttk.Progressbar(
            server_frame,
            mode="determinate",
            length=160,
            maximum=100,
            style="Stopped.Horizontal.TProgressbar",
        )
        self.server_progress.pack(side=self.tk.LEFT, padx=8, fill=self.tk.X, expand=False)
        self.server_progress["value"] = 100
        self.server_status_label = self.ttk.Label(server_frame, textvariable=self.server_status_var)
        self.server_status_label.pack(side=self.tk.LEFT)

        self.custom_frame = self.ttk.LabelFrame(frame, text="Configuración personalizada", padding=6)
        self.custom_frame.grid(row=4, column=0, columnspan=5, sticky="we", pady=(8, 0))
        self.custom_frame.columnconfigure(1, weight=1)
        self.custom_frame.columnconfigure(3, weight=1)

        self.ttk.Label(self.custom_frame, text="Modelo GGUF:").grid(row=0, column=0, sticky="w")
        self.custom_model_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_model_var, width=60)
        self.custom_model_entry.grid(row=0, column=1, sticky="we", padx=4)
        browse_btn = self.ttk.Button(self.custom_frame, text="Examinar…", command=self._browse_model)
        browse_btn.grid(row=0, column=2, padx=4)

        self.ttk.Label(self.custom_frame, text="ctx-size").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ctx_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_ctx_var, width=10)
        ctx_entry.grid(row=1, column=1, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="batch-size").grid(row=1, column=2, sticky="w", pady=(4, 0))
        batch_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_batch_var, width=10)
        batch_entry.grid(row=1, column=3, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="n_gpu_layers").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ngl_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_ngl_var, width=10)
        ngl_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="threads").grid(row=2, column=2, sticky="w", pady=(4, 0))
        threads_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_threads_var, width=10)
        threads_entry.grid(row=2, column=3, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="tensor-split").grid(row=3, column=0, sticky="w", pady=(4, 0))
        tensor_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_tensor_var, width=18)
        tensor_entry.grid(row=3, column=1, sticky="we", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="rope_freq_base").grid(row=3, column=2, sticky="w", pady=(4, 0))
        rope_base_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_rope_base_var, width=10)
        rope_base_entry.grid(row=3, column=3, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="rope_freq_scale").grid(row=4, column=0, sticky="w", pady=(4, 0))
        rope_scale_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_rope_scale_var, width=10)
        rope_scale_entry.grid(row=4, column=1, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="top_k").grid(row=4, column=2, sticky="w", pady=(4, 0))
        topk_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_top_k_var, width=10)
        topk_entry.grid(row=4, column=3, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="temperature").grid(row=5, column=0, sticky="w", pady=(4, 0))
        temp_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_temp_var, width=10)
        temp_entry.grid(row=5, column=1, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="top_p").grid(row=5, column=2, sticky="w", pady=(4, 0))
        topp_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_top_p_var, width=10)
        topp_entry.grid(row=5, column=3, sticky="w", pady=(4, 0))

        self.ttk.Label(self.custom_frame, text="repeat_penalty").grid(row=6, column=0, sticky="w", pady=(4, 0))
        repeat_entry = self.ttk.Entry(self.custom_frame, textvariable=self.custom_repeat_var, width=10)
        repeat_entry.grid(row=6, column=1, sticky="w", pady=(4, 0))

        flash_check = self.ttk.Checkbutton(self.custom_frame, text="Usar Flash Attention", variable=self.custom_flash_var)
        flash_check.grid(row=6, column=2, columnspan=2, sticky="w", pady=(4, 0))

        btn_frame = self.ttk.Frame(self.custom_frame)
        btn_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.ttk.Button(btn_frame, text="Autoconfigurar", command=self._on_custom_autoconfig).pack(side=self.tk.LEFT, padx=(0, 6))
        self.ttk.Button(btn_frame, text="Guardar ajustes", command=self._on_custom_save).pack(side=self.tk.LEFT)
        self.custom_frame.grid_remove()

        endpoint_opts_frame = self.ttk.LabelFrame(frame, text="Parámetros por endpoint", padding=6)
        endpoint_opts_frame.grid(row=5, column=0, columnspan=5, sticky="we", pady=(8, 0))
        endpoint_opts_frame.columnconfigure(0, weight=1)
        self.endpoint_tabs = self.ttk.Notebook(endpoint_opts_frame)
        self.endpoint_tabs.pack(fill=self.tk.BOTH, expand=True)
        self.endpoint_tabs.configure(width=860, height=180)
        for endpoint in ["llm", "clm", "vlm", "alm", "slm"]:
            tab = self.ttk.Frame(self.endpoint_tabs)
            tab.columnconfigure(1, weight=1)
            self.endpoint_tabs.add(tab, text=endpoint.upper())
            settings = self.profile_manager.get_endpoint_settings(endpoint)
            vars_for_endpoint: Dict[str, object] = {}
            for row, (key, value) in enumerate(settings.items()):
                self.ttk.Label(tab, text=key).grid(row=row, column=0, sticky="w", padx=4, pady=2)
                var = self.tk.StringVar(value=str(value))
                entry = self.ttk.Entry(tab, textvariable=var, width=12)
                entry.grid(row=row, column=1, sticky="we", padx=4, pady=2)
                vars_for_endpoint[key] = var
            self.endpoint_settings_vars[endpoint] = vars_for_endpoint
        self.ttk.Button(endpoint_opts_frame, text="Guardar parámetros", command=self._save_endpoint_settings).pack(
            anchor="w", pady=(6, 0)
        )

    def _build_notebook(self):
        notebook = self.ttk.Notebook(self.root)
        notebook.pack(expand=True, fill=self.tk.BOTH, padx=10, pady=10)

        self.logs_tab = self.ttk.Frame(notebook, padding=10)
        self.feedback_tab = self.ttk.Frame(notebook, padding=10)
        self.suggestions_tab = self.ttk.Frame(notebook, padding=10)

        notebook.add(self.logs_tab, text="Logs")
        notebook.add(self.feedback_tab, text="Feedback testers")
        notebook.add(self.suggestions_tab, text="Recomendaciones")

        self._build_logs_tab()
        self._build_feedback_tab()
        self._build_suggestions_tab()

    def _build_logs_tab(self):
        btn_frame = self.ttk.Frame(self.logs_tab)
        btn_frame.pack(fill=self.tk.X)

        self.ttk.Button(btn_frame, text="Ver llama-server.log", command=lambda: self._load_log("llama-server.log")).pack(
            side=self.tk.LEFT, padx=4
        )
        self.ttk.Button(btn_frame, text="Ver lmdeploy.log", command=lambda: self._load_log("lmdeploy.log")).pack(
            side=self.tk.LEFT, padx=4
        )

        self.log_widget = self.ScrolledText(self.logs_tab, wrap="word")
        self.log_widget.pack(expand=True, fill=self.tk.BOTH, pady=6)
        self.log_widget.configure(font=("Consolas", 10), background="#fafbff", relief="flat", borderwidth=1)

    def _build_feedback_tab(self):
        form = self.ttk.Frame(self.feedback_tab)
        form.pack(fill=self.tk.X, pady=6)

        self.ttk.Label(form, text="Resultado:").grid(row=0, column=0, sticky="w")
        issue_box = self.ttk.Combobox(form, values=["ok", "freeze", "oom", "otro"], textvariable=self.feedback_issue_var)
        issue_box.grid(row=0, column=1, sticky="w", padx=6)
        issue_box.current(0)

        self.ttk.Label(form, text="Notas:").grid(row=1, column=0, sticky="nw")
        self.feedback_text = self.ScrolledText(form, height=6, wrap="word")
        self.feedback_text.grid(row=1, column=1, columnspan=3, sticky="we")
        self.feedback_text.configure(font=("Segoe UI", 10), background="#ffffff", relief="solid", borderwidth=1)

        save_btn = self.ttk.Button(form, text="Guardar feedback", command=self._save_feedback)
        save_btn.grid(row=2, column=1, sticky="w", pady=6)

        summary_frame = self.ttk.LabelFrame(self.feedback_tab, text="Resumen de registros", padding=6)
        summary_frame.pack(fill=self.tk.BOTH, expand=True, pady=(8, 0))

        self.feedback_summary_widget = self.ScrolledText(summary_frame, height=6, wrap="word")
        self.feedback_summary_widget.pack(fill=self.tk.BOTH, expand=True)
        self.feedback_summary_widget.configure(font=("Segoe UI", 10), background="#ffffff", relief="flat", borderwidth=1)

    def _build_suggestions_tab(self):
        self.suggestions_widget = self.ScrolledText(self.suggestions_tab, wrap="word")
        self.suggestions_widget.pack(fill=self.tk.BOTH, expand=True)
        self.suggestions_widget.configure(font=("Segoe UI", 10), background="#ffffff", relief="flat", borderwidth=1)

    # --- Event handlers ---------------------------------------------
    def _on_apply_profile(self):
        label = self.preset_box.get()
        key = self._preset_key_by_label.get(label)
        if not key:
            self.status_var.set("Seleccione un perfil válido.")
            return
        ok, msg = self.profile_manager.set_active_profile(key, force=self.force_var.get())
        self.status_var.set(msg)
        if not ok:
            self.messagebox.showwarning("No se pudo aplicar el perfil", msg)
        else:
            self.profile_var.set(key)
            self._refresh_profile_summary()
            self._refresh_presets()
            if key == "personalizado":
                self._populate_custom_fields()
                self._show_custom_frame(True)

    def _refresh_profile_summary(self):
        summary = self.profile_manager.describe_active_profile()
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        self.profile_summary.delete("1.0", self.tk.END)
        self.profile_summary.insert(self.tk.END, text)
        self._refresh_server_status()

    def _show_custom_frame(self, show: bool):
        if not hasattr(self, "custom_frame"):
            return
        if show and not self._custom_visible:
            self.custom_frame.grid()
            self._custom_visible = True
        elif not show and self._custom_visible:
            self.custom_frame.grid_remove()
            self._custom_visible = False

    def _populate_custom_fields(self):
        try:
            cfg = self.profile_manager.get_custom_settings()
        except Exception as exc:
            self.status_var.set(f"Error leyendo ajustes personalizados: {exc}")
            return
        self.custom_model_var.set(cfg.get("model_path", ""))
        self.custom_ctx_var.set(str(cfg.get("ctx_size", "")))
        self.custom_ngl_var.set(str(cfg.get("n_gpu_layers", "")))
        self.custom_threads_var.set(str(cfg.get("threads", "")))
        self.custom_batch_var.set(str(cfg.get("batch_size", "")))
        self.custom_tensor_var.set(cfg.get("tensor_split", ""))
        self.custom_rope_base_var.set(str(cfg.get("rope_freq_base", "")))
        self.custom_rope_scale_var.set(str(cfg.get("rope_freq_scale", "")))
        self.custom_temp_var.set(str(cfg.get("temperature", "")))
        self.custom_top_p_var.set(str(cfg.get("top_p", "")))
        self.custom_top_k_var.set(str(cfg.get("top_k", "")))
        self.custom_repeat_var.set(str(cfg.get("repeat_penalty", "")))
        self.custom_flash_var.set(bool(cfg.get("flash_attn", False)))

    def _collect_custom_updates(self) -> Dict[str, str]:
        return {
            "model_path": self.custom_model_var.get().strip(),
            "ctx_size": self.custom_ctx_var.get().strip(),
            "n_gpu_layers": self.custom_ngl_var.get().strip(),
            "threads": self.custom_threads_var.get().strip(),
            "batch_size": self.custom_batch_var.get().strip(),
            "tensor_split": self.custom_tensor_var.get().strip(),
            "rope_freq_base": self.custom_rope_base_var.get().strip(),
            "rope_freq_scale": self.custom_rope_scale_var.get().strip(),
            "temperature": self.custom_temp_var.get().strip(),
            "top_p": self.custom_top_p_var.get().strip(),
            "top_k": self.custom_top_k_var.get().strip(),
            "repeat_penalty": self.custom_repeat_var.get().strip(),
            "flash_attn": self.custom_flash_var.get(),
        }

    def _sync_endpoint_checkboxes(self):
        cfg = self.profile_manager.get_endpoint_config()
        self._endpoint_updating = True
        try:
            for name, var in self.endpoint_vars.items():
                var.set(cfg.get(name, False))
        finally:
            self._endpoint_updating = False

    def _populate_endpoint_settings(self):
        for endpoint, fields in self.endpoint_settings_vars.items():
            settings = self.profile_manager.get_endpoint_settings(endpoint)
            for key, var in fields.items():
                var.set(str(settings.get(key, "")))

    def _save_endpoint_settings(self):
        errors: list[str] = []
        for endpoint, fields in self.endpoint_settings_vars.items():
            updates = {key: var.get().strip() for key, var in fields.items()}
            ok, msg = self.profile_manager.update_endpoint_settings(endpoint, updates)
            if not ok:
                errors.append(msg)
        if errors:
            self.messagebox.showerror("Parámetros por endpoint", "\n".join(errors))
        else:
            self.status_var.set("Parámetros de endpoints guardados.")
        self._populate_endpoint_settings()

    def _gather_enabled_endpoints(self) -> list[str]:
        return [name for name, var in self.endpoint_vars.items() if var.get()]

    def _on_endpoint_toggle(self, endpoint: str):
        if self._endpoint_updating:
            return
        try:
            value = self.endpoint_vars[endpoint].get()
            if endpoint == "llm" and not value:
                if self.endpoint_vars.get("alm", self.tk.BooleanVar(value=False)).get() or self.endpoint_vars.get("slm", self.tk.BooleanVar(value=False)).get():
                    self.messagebox.showwarning(
                        "Dependencia",
                        "Los endpoints ALM/SLM requieren LLM activo. No se puede desactivar LLM mientras estén habilitados.",
                    )
                    self._endpoint_updating = True
                    self.endpoint_vars[endpoint].set(True)
                    self._endpoint_updating = False
                    return
            if endpoint in {"alm", "slm"} and value and not self.endpoint_vars.get("llm", self.tk.BooleanVar(value=False)).get():
                self._endpoint_updating = True
                self.endpoint_vars["llm"].set(True)
                self._endpoint_updating = False
                self.profile_manager.update_endpoint_config({"llm": True})
            self.profile_manager.update_endpoint_config({endpoint: value})
        finally:
            self._endpoint_updating = False
        self._sync_endpoint_checkboxes()
        self._refresh_server_status()

    def _browse_model(self):
        filename = self.filedialog.askopenfilename(
            title="Seleccionar modelo GGUF",
            filetypes=[("GGUF", "*.gguf"), ("Todos", "*.*")],
        )
        if filename:
            self.custom_model_var.set(filename)

    def _set_server_busy(self, busy: bool, message: Optional[str] = None):
        self._server_busy = busy
        if busy:
            self.server_button.configure(state="disabled")
            self.server_progress.configure(mode="indeterminate", style="Busy.Horizontal.TProgressbar")
            self.server_progress.start(10)
            if message:
                self.server_status_var.set(message)
        else:
            self.server_progress.stop()
            self.server_button.configure(state="normal")
            self.server_progress.configure(mode="determinate")
            if message:
                self.server_status_var.set(message)

    def _on_server_toggle(self):
        if not self.manager:
            self.messagebox.showwarning("Servidor", "Manager no disponible.")
            return
        if self._server_thread and self._server_thread.is_alive():
            return
        running = self.manager.is_running("llm")
        if running:
            self._set_server_busy(True, "Deteniendo servidor...")
            self._server_thread = threading.Thread(target=self._stop_server_worker, daemon=True)
            self._server_thread.start()
        else:
            profile_info = self.profile_manager.describe_active_profile()
            model_path = profile_info.get("model", "")
            if not model_path or (self.profile_manager.active_profile.user_configurable and not Path(model_path).exists()):
                self.messagebox.showwarning("Servidor", "Definí un modelo válido antes de iniciar el servidor.")
                return
            self._set_server_busy(True, "Iniciando servidor...")
            self.status_var.set("Iniciando servidor y precargando modelos…")
            self.server_button.configure(text="Iniciando…")
            self._server_thread = threading.Thread(target=self._start_server_worker, daemon=True)
            self._server_thread.start()

    def _start_server_worker(self):
        warm_warning = None
        error = None
        enabled = self._gather_enabled_endpoints()
        try:
            if self.warm_callback:
                self.warm_callback(enabled)
        except Exception as exc:
            warm_warning = exc
        need_llm = any(name in ("llm", "alm", "slm") for name in enabled)
        if not need_llm:
            self.root.after(0, lambda: self._after_warm_only(warm_warning))
            return
        try:
            self.manager.ensure_mode("llm")
        except Exception as exc:
            error = exc
        self.root.after(0, lambda: self._after_server_action(start=True, error=error, warm_warning=warm_warning))

    def _stop_server_worker(self):
        error = None
        try:
            self.manager.stop()
        except Exception as exc:
            error = exc
        self.root.after(0, lambda: self._after_server_action(start=False, error=error, warm_warning=None))

    def _after_warm_only(self, warm_warning: Optional[Exception]):
        self._set_server_busy(False)
        if warm_warning:
            self.status_var.set(f"Warm-up completado con advertencia: {warm_warning}")
        else:
            self.status_var.set("Recursos precargados. No se inició servidor LLM.")
        self._refresh_server_status()

    def _after_server_action(self, start: bool, error: Optional[Exception], warm_warning: Optional[Exception] = None):
        self._set_server_busy(False)
        if error:
            message = f"Error {'iniciando' if start else 'deteniendo'} servidor: {error}"
            self.server_status_var.set(message)
            self.messagebox.showerror("Servidor LLM", str(error))
        else:
            if warm_warning:
                self.status_var.set(f"Warm-up con advertencia: {warm_warning}")
            elif start:
                self.status_var.set("Servidor LLM listo.")
            else:
                self.status_var.set("Servidor LLM detenido.")
        self._refresh_server_status()

    def _refresh_server_status(self):
        if not self.manager:
            self.server_status_var.set("Servidor LLM no disponible")
            self.server_button.configure(state="disabled")
            return
        if self._server_busy:
            return
        running = self.manager.is_running("llm")
        if running:
            self.server_button.configure(text="Matar servidor", state="normal")
            self.server_status_var.set("Servidor LLM: en ejecución")
            self.server_progress.configure(mode="determinate", style="Running.Horizontal.TProgressbar")
            self.server_progress["value"] = 100
        else:
            self.server_button.configure(text="Iniciar servidor", state="normal")
            self.server_status_var.set("Servidor LLM: detenido")
            self.server_progress.configure(mode="determinate", style="Stopped.Horizontal.TProgressbar")
            self.server_progress["value"] = 100
        self.server_progress.stop()

    def _on_custom_autoconfig(self):
        try:
            cfg = self.profile_manager.autoconfigure_custom()
        except Exception as exc:
            self.messagebox.showerror("Autoconfigurar", f"No se pudo autoconfigurar: {exc}")
            return
        self._populate_custom_fields()
        self._refresh_profile_summary()
        self._refresh_presets()
        msg = (
            "Perfil personalizado autoconfigurado según el hardware. "
            "Modelo actual: {}".format(cfg.get("model_path", ""))
        )
        self.status_var.set(msg)
        self._refresh_server_status()

    def _on_custom_save(self):
        updates = self._collect_custom_updates()
        ok, msg = self.profile_manager.update_custom_settings(updates)
        if not ok:
            self.messagebox.showerror("Guardar ajustes", msg)
            return
        self.status_var.set(msg)
        self._populate_custom_fields()
        self._refresh_profile_summary()
        self._refresh_presets()
        self._refresh_server_status()

    def _on_preset_selected(self, _event=None):
        label = self.preset_box.get()
        key = self._preset_key_by_label.get(label)
        is_custom = key == "personalizado"
        self._show_custom_frame(is_custom)
        if is_custom:
            self._populate_custom_fields()
        self._refresh_server_status()

    def _refresh_presets(self):
        current_selection = self.preset_box.get()
        presets = self.profile_manager.list_presets()
        preset_options = []
        self._preset_key_by_label.clear()
        for preset in presets:
            label = f"{preset['title']} ({preset['key']}) - {preset['status']}"
            preset_options.append(label)
            self._preset_key_by_label[label] = preset["key"]
        self.preset_box["values"] = preset_options
        target_label = None
        if current_selection in self._preset_key_by_label:
            target_label = current_selection
        elif preset_options:
            target_label = next(
                (label for label, key in self._preset_key_by_label.items() if key == self.profile_var.get()),
                preset_options[0],
            )
        if target_label:
            self.preset_box.set(target_label)
        current_key = self._preset_key_by_label.get(self.preset_box.get())
        show_custom = current_key == "personalizado"
        self._show_custom_frame(show_custom)
        if show_custom:
            self._populate_custom_fields()
        self._sync_endpoint_checkboxes()
        self._populate_endpoint_settings()
        self._refresh_server_status()

    def _load_log(self, name: str):
        path = self.log_dir / name
        if not path.exists():
            self.log_widget.delete("1.0", self.tk.END)
            self.log_widget.insert(self.tk.END, f"No se encontró {name}")
            return
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            content = f"Error leyendo {name}: {exc}"
        self.log_widget.delete("1.0", self.tk.END)
        self.log_widget.insert(self.tk.END, content[-8000:])

    def _save_feedback(self):
        notes = self.feedback_text.get("1.0", self.tk.END).strip()
        if not notes:
            self.messagebox.showinfo("Feedback", "Ingrese alguna nota para guardar.")
            return
        issue = self.feedback_issue_var.get()
        self.profile_manager.append_feedback(self.profile_var.get(), notes, issue)
        self.feedback_text.delete("1.0", self.tk.END)
        self.status_var.set("Feedback guardado. ¡Gracias!")
        self._refresh_feedback_summary()

    def _refresh_feedback_summary(self):
        stats = self.profile_manager.get_feedback_summary()
        lines = [f"{key}: {count}" for key, count in stats.items()]
        if not lines:
            lines = ["Sin registros todavía."]
        text = "\n".join(lines)
        self.feedback_summary_widget.delete("1.0", self.tk.END)
        self.feedback_summary_widget.insert(self.tk.END, text)

    def _refresh_suggestions(self):
        suggestions = self.preflight_report.get("warnings", []) + self.preflight_report.get("suggestions", [])
        if not suggestions:
            suggestions = ["Sistema listo. No hay observaciones pendientes."]
        text = "\n".join(f"- {item}" for item in suggestions)
        self.suggestions_widget.delete("1.0", self.tk.END)
        self.suggestions_widget.insert(self.tk.END, text)

    def _periodic_update(self):
        self._refresh_profile_summary()
        self._sync_endpoint_checkboxes()
        self._refresh_server_status()
        self.root.after(10000, self._periodic_update)

    def _schedule_updates(self):
        self.root.after(10000, self._periodic_update)

    def run(self):
        self.root.mainloop()


def launch_gui(
    profile_manager,
    preflight_report: Dict,
    log_dir: Path,
    logo_path: Optional[Path] = None,
    manager=None,
    warm_callback=None,
    splash_path: Optional[Path] = None,
):
    gui = StudioGUI(
        profile_manager,
        preflight_report,
        log_dir,
        logo_path=logo_path,
        manager=manager,
        warm_callback=warm_callback,
        splash_path=splash_path,
    )
    gui.run()


def start_gui_thread(
    profile_manager,
    preflight_report: Dict,
    log_dir: Path,
    logo_path: Optional[Path] = None,
    manager=None,
    warm_callback=None,
    splash_path: Optional[Path] = None,
) -> Optional[threading.Thread]:
    try:
        thread = threading.Thread(
            target=launch_gui,
            args=(profile_manager, preflight_report, log_dir, logo_path, manager, warm_callback, splash_path),
            daemon=True,
            name="unlz_gui",
        )
        thread.start()
        return thread
    except Exception as exc:
        print(f"[gui] No se pudo iniciar la interfaz: {exc}")
        return None
