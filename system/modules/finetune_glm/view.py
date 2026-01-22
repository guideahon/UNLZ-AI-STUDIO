import os
import sys
import threading
import subprocess
import webbrowser
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from modules.base import StudioModule


class FinetuneGLMModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "finetune_glm", "GLM-4.7 Fine-tune")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = FinetuneGLMView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class FinetuneGLMView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tr = app.tr
        self._busy = False

        self.app_root = Path(__file__).resolve().parents[3]
        self.output_dir = self.app_root / "system" / "finetune-out"
        self.script_path = self.app_root / "system" / "data" / "finetune_glm" / "finetune_glm_4_7_flash.py"

        self.build_ui()

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=self.tr("finetune_title"), font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text=self.tr("finetune_subtitle"), text_color="gray").pack(anchor="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkButton(actions, text=self.tr("finetune_btn_install"), command=self.install_deps).pack(side="left", padx=5)
        ctk.CTkButton(actions, text=self.tr("finetune_btn_open_docs"), command=self.open_docs).pack(side="left", padx=5)
        ctk.CTkButton(actions, text=self.tr("finetune_btn_open_output"), command=self.open_output).pack(side="left", padx=5)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(status_frame, text=self.tr("finetune_status_label")).pack(side="left")
        self.status_value = ctk.CTkLabel(status_frame, text=self.tr("finetune_status_idle"), text_color="gray")
        self.status_value.pack(side="left", padx=(6, 0))

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        # Dataset
        ctk.CTkLabel(body, text=self.tr("finetune_dataset_title"), font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 0)
        )
        ctk.CTkLabel(body, text=self.tr("finetune_dataset_label")).grid(row=1, column=0, sticky="w", padx=10, pady=(6, 0))
        self.dataset_entry = ctk.CTkEntry(body, placeholder_text=self.tr("finetune_dataset_placeholder"))
        self.dataset_entry.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkButton(body, text=self.tr("finetune_btn_browse"), command=self.pick_dataset).grid(
            row=3, column=0, sticky="w", padx=10, pady=(0, 10)
        )

        ctk.CTkLabel(body, text=self.tr("finetune_base_model")).grid(row=4, column=0, sticky="w", padx=10, pady=(0, 0))
        base_entry = ctk.CTkEntry(body)
        base_entry.insert(0, "unsloth/GLM-4.7-Flash")
        base_entry.configure(state="disabled")
        base_entry.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))

        ctk.CTkLabel(body, text=self.tr("finetune_format_label")).grid(row=6, column=0, sticky="w", padx=10, pady=(0, 0))
        self.format_menu = ctk.CTkOptionMenu(body, values=[self.tr("finetune_format_sharegpt")])
        self.format_menu.grid(row=7, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Output / export
        ctk.CTkLabel(body, text=self.tr("finetune_output_title"), font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=1, sticky="w", padx=10, pady=(10, 0)
        )
        ctk.CTkLabel(body, text=self.tr("finetune_output_label")).grid(row=1, column=1, sticky="w", padx=10, pady=(6, 0))
        self.output_entry = ctk.CTkEntry(body, placeholder_text=str(self.output_dir))
        self.output_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkButton(body, text=self.tr("finetune_btn_browse"), command=self.pick_output).grid(
            row=3, column=1, sticky="w", padx=10, pady=(0, 10)
        )

        ctk.CTkLabel(body, text=self.tr("finetune_gguf_label")).grid(row=4, column=1, sticky="w", padx=10, pady=(0, 0))
        self.gguf_var = ctk.StringVar(value="yes")
        self.gguf_menu = ctk.CTkOptionMenu(body, values=["yes", "no"], variable=self.gguf_var)
        self.gguf_menu.grid(row=5, column=1, sticky="ew", padx=10, pady=(0, 6))

        ctk.CTkLabel(body, text=self.tr("finetune_gguf_quant_label")).grid(row=6, column=1, sticky="w", padx=10, pady=(0, 0))
        self.gguf_quant = ctk.CTkOptionMenu(body, values=["q4_k_m", "q5_k_m", "q8_0", "f16"])
        self.gguf_quant.grid(row=7, column=1, sticky="ew", padx=10, pady=(0, 10))

        # Training settings
        ctk.CTkLabel(body, text=self.tr("finetune_train_title"), font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=8, column=0, sticky="w", padx=10, pady=(10, 0)
        )
        self._add_labeled_entry(body, self.tr("finetune_epochs"), "1.0", 9, 0)
        self._add_labeled_entry(body, self.tr("finetune_batch_size"), "1", 10, 0)
        self._add_labeled_entry(body, self.tr("finetune_grad_accum"), "8", 11, 0)
        self._add_labeled_entry(body, self.tr("finetune_learning_rate"), "0.0002", 12, 0)
        self._add_labeled_entry(body, self.tr("finetune_max_seq"), "4096", 13, 0)
        self._add_labeled_entry(body, self.tr("finetune_lora_r"), "16", 14, 0)
        self._add_labeled_entry(body, self.tr("finetune_lora_alpha"), "16", 15, 0)
        self._add_labeled_entry(body, self.tr("finetune_lora_dropout"), "0.0", 16, 0)

        run_box = ctk.CTkFrame(self, fg_color="transparent")
        run_box.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(run_box, text=self.tr("finetune_btn_run"), command=self.run_finetune).pack(side="left", padx=5)

    def _add_labeled_entry(self, parent, label, default, row, column):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=column, sticky="w", padx=10, pady=(4, 0))
        entry = ctk.CTkEntry(parent)
        entry.insert(0, default)
        entry.grid(row=row, column=column, sticky="ew", padx=10, pady=(0, 4))
        setattr(self, f"entry_{row}_{column}", entry)

    def pick_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON/JSONL", "*.json;*.jsonl"), ("All files", "*.*")])
        if file_path:
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, file_path)

    def pick_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder)

    def open_docs(self):
        webbrowser.open("https://unsloth.ai/docs/models/glm-4.7-flash")

    def open_output(self):
        out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(out_dir))

    def install_deps(self):
        packages = [
            "unsloth",
            "transformers==5.0.0rc3",
            "datasets",
            "trl",
            "peft",
            "accelerate",
        ]
        self._run_process(
            [sys.executable.replace("pythonw.exe", "python.exe"), "-m", "pip", "install", "--pre", *packages]
        )

    def run_finetune(self):
        dataset = self.dataset_entry.get().strip()
        if not dataset:
            messagebox.showwarning(self.tr("status_error"), self.tr("finetune_msg_missing_dataset"))
            return
        if not self.script_path.exists():
            messagebox.showerror(self.tr("status_error"), f"Script not found: {self.script_path}")
            return

        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        args = [
            sys.executable.replace("pythonw.exe", "python.exe"),
            "-u",
            str(self.script_path),
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
            "--max-seq-len",
            self._get_entry_value(13, 0, "4096"),
            "--epochs",
            self._get_entry_value(9, 0, "1.0"),
            "--batch-size",
            self._get_entry_value(10, 0, "1"),
            "--grad-accum",
            self._get_entry_value(11, 0, "8"),
            "--learning-rate",
            self._get_entry_value(12, 0, "0.0002"),
            "--lora-r",
            self._get_entry_value(14, 0, "16"),
            "--lora-alpha",
            self._get_entry_value(15, 0, "16"),
            "--lora-dropout",
            self._get_entry_value(16, 0, "0.0"),
        ]

        if self.gguf_var.get() == "yes":
            args += ["--export-gguf", "--gguf-quant", self.gguf_quant.get()]

        self._run_process(args)

    def _get_output_dir(self) -> Path:
        value = self.output_entry.get().strip()
        return Path(value) if value else self.output_dir

    def _get_entry_value(self, row, column, fallback):
        entry = getattr(self, f"entry_{row}_{column}", None)
        if not entry:
            return fallback
        value = entry.get().strip()
        return value if value else fallback

    def _set_busy(self, busy):
        self._busy = busy
        if self.status_value:
            self.status_value.configure(
                text=self.tr("status_in_progress") if busy else self.tr("finetune_status_idle"),
                text_color="orange" if busy else "gray",
            )

    def _run_process(self, cmd):
        if self._busy:
            return

        def worker():
            self._set_busy(True)
            logging.info("Finetune command: %s", " ".join(cmd))
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.app_root / "system"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                if process.stdout:
                    for line in process.stdout:
                        logging.info(line.rstrip())
                process.wait()
                logging.info("Finetune finished with code %s", process.returncode)
            except Exception as exc:
                logging.error("Finetune error: %s", exc)
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()
