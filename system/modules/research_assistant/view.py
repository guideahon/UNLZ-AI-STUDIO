import customtkinter as ctk
import os
import re
import json
import math
import shutil
import threading
from pathlib import Path
from uuid import uuid4
from tkinter import filedialog, messagebox
import requests

from modules.base import StudioModule

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella", "ellas",
    "ellos", "en", "era", "eramos", "eran", "eres", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estaba", "estaban", "estado", "estais", "estamos", "estan", "estar",
    "estas", "este", "esto", "estos", "fue", "fuera", "fueron", "ha", "hace", "haces",
    "hacia", "han", "hasta", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi",
    "mis", "mucho", "muy", "no", "nos", "nosotros", "o", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "quien", "se", "sea",
    "ser", "si", "sin", "sobre", "son", "su", "sus", "tambien", "tan", "tanto", "te",
    "tengo", "ti", "tiene", "tienen", "tu", "tus", "un", "una", "uno", "unos", "y"
}


class ResearchAssistantModule(StudioModule):
    def __init__(self, parent):
        super().__init__(parent, "research_assistant", "Asistente de Investigacion")
        self.view = None
        self.app = parent

    def get_view(self) -> ctk.CTkFrame:
        self.view = ResearchAssistantView(self.app.main_container, self.app)
        return self.view

    def on_enter(self):
        pass

    def on_leave(self):
        pass


class ResearchAssistantView(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app

        self.data_dir = Path(__file__).resolve().parents[2] / "data" / "research_assistant"
        self.docs_dir = self.data_dir / "docs"
        self.library_path = self.data_dir / "library.json"
        self.index_path = self.data_dir / "index.json"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.library = self.load_library()
        self.index = self.load_index()
        self.idf = self.build_idf(self.index)

        self.selected_doc_id = None

        tr = self.app.tr

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_library = self.tabview.add(tr("tab_library"))
        self.tab_ask = self.tabview.add(tr("tab_ask"))

        self.build_library_tab()
        self.build_ask_tab()

    # --- Library UI ---
    def build_library_tab(self):
        tr = self.app.tr
        frame = self.tab_library
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        actions = ctk.CTkFrame(frame, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkButton(actions, text=tr("btn_add_pdf"), command=self.add_pdf).pack(side="left", padx=5)
        ctk.CTkButton(actions, text=tr("btn_remove_doc"), command=self.remove_selected).pack(side="left", padx=5)
        ctk.CTkButton(actions, text=tr("btn_build_index"), command=self.build_index).pack(side="left", padx=5)

        self.library_status = ctk.CTkLabel(actions, text="", text_color="gray")
        self.library_status.pack(side="right", padx=10)

        body = ctk.CTkFrame(frame, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.docs_scroll = ctk.CTkScrollableFrame(body, label_text=tr("lbl_docs"))
        self.docs_scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        details = ctk.CTkFrame(body)
        details.grid(row=0, column=1, sticky="nsew")
        details.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(details, text=tr("lbl_metadata"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        self.title_var = ctk.StringVar()
        self.authors_var = ctk.StringVar()
        self.year_var = ctk.StringVar()
        self.venue_var = ctk.StringVar()
        self.url_var = ctk.StringVar()

        self._add_meta_row(details, tr("lbl_title"), self.title_var, 1)
        self._add_meta_row(details, tr("lbl_authors"), self.authors_var, 2)
        self._add_meta_row(details, tr("lbl_year"), self.year_var, 3)
        self._add_meta_row(details, tr("lbl_venue"), self.venue_var, 4)
        self._add_meta_row(details, tr("lbl_url"), self.url_var, 5)

        ctk.CTkButton(details, text=tr("btn_save_meta"), command=self.save_metadata).grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(details, text=tr("lbl_summary"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
        self.summary_box = ctk.CTkTextbox(details, height=120)
        self.summary_box.grid(row=8, column=0, columnspan=2, sticky="nsew", padx=10)

        ctk.CTkButton(details, text=tr("btn_generate_summary"), command=self.generate_summary).grid(row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(details, text=tr("lbl_citations"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=10, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
        self.citations_box = ctk.CTkTextbox(details, height=90)
        self.citations_box.grid(row=11, column=0, columnspan=2, sticky="nsew", padx=10)

        cite_buttons = ctk.CTkFrame(details, fg_color="transparent")
        cite_buttons.grid(row=12, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
        ctk.CTkButton(cite_buttons, text=tr("btn_copy_apa"), command=lambda: self.copy_citation("apa")).pack(side="left", padx=5)
        ctk.CTkButton(cite_buttons, text=tr("btn_copy_ieee"), command=lambda: self.copy_citation("ieee")).pack(side="left", padx=5)

        self.refresh_doc_list()

    def _add_meta_row(self, parent, label, var, row):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=4)
        ctk.CTkEntry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=10, pady=4)

    # --- Ask UI ---
    def build_ask_tab(self):
        tr = self.app.tr
        frame = self.tab_ask
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        settings = ctk.CTkFrame(frame, fg_color="transparent")
        settings.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(settings, text=tr("lbl_endpoint")).pack(side="left", padx=5)
        endpoint_default = self.app.get_setting("rag_endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        self.endpoint_var = ctk.StringVar(value=endpoint_default)
        endpoint_entry = ctk.CTkEntry(settings, textvariable=self.endpoint_var, width=420)
        endpoint_entry.pack(side="left", padx=5)
        ctk.CTkButton(settings, text=tr("btn_save_endpoint"), command=self.save_endpoint).pack(side="left", padx=5)

        ask_row = ctk.CTkFrame(frame, fg_color="transparent")
        ask_row.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.question_entry = ctk.CTkEntry(ask_row, placeholder_text=tr("placeholder_question"))
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(ask_row, text=tr("btn_ask"), command=self.ask_question).pack(side="left")

        output = ctk.CTkFrame(frame, fg_color="transparent")
        output.grid(row=2, column=0, sticky="nsew")
        output.grid_columnconfigure(0, weight=1)
        output.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(output, text=tr("lbl_answer"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, sticky="w", padx=5)
        self.answer_box = ctk.CTkTextbox(output)
        self.answer_box.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 10))

        ctk.CTkLabel(output, text=tr("lbl_sources"), font=ctk.CTkFont(size=16, weight="bold")).grid(row=2, column=0, sticky="w", padx=5)
        self.sources_box = ctk.CTkTextbox(output, height=120)
        self.sources_box.grid(row=3, column=0, sticky="nsew", padx=5, pady=(0, 10))

        self.ask_status = ctk.CTkLabel(frame, text="", text_color="gray")
        self.ask_status.grid(row=4, column=0, sticky="w", padx=5)

    # --- Library Actions ---
    def refresh_doc_list(self):
        tr = self.app.tr
        for widget in self.docs_scroll.winfo_children():
            widget.destroy()

        if not self.library:
            ctk.CTkLabel(self.docs_scroll, text=tr("status_no_docs"), text_color="gray").pack(pady=10)
            return

        for doc in self.library:
            title = doc.get("title") or doc.get("original_name") or doc.get("filename")
            btn = ctk.CTkButton(self.docs_scroll, text=title, anchor="w", command=lambda d=doc["id"]: self.select_doc(d))
            btn.pack(fill="x", pady=2, padx=5)

    def select_doc(self, doc_id):
        doc = self.get_doc(doc_id)
        if not doc:
            return
        self.selected_doc_id = doc_id
        self.title_var.set(doc.get("title", ""))
        self.authors_var.set(doc.get("authors", ""))
        self.year_var.set(doc.get("year", ""))
        self.venue_var.set(doc.get("venue", ""))
        self.url_var.set(doc.get("url", ""))

        self.summary_box.delete("0.0", "end")
        summary = doc.get("summary", "")
        if summary:
            self.summary_box.insert("end", summary)

        self.update_citations(doc)

    def add_pdf(self):
        tr = self.app.tr
        if not PyPDF2:
            messagebox.showerror(tr("status_error"), tr("msg_pdf_dependency"))
            return

        file_path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not file_path:
            return

        try:
            src = Path(file_path)
            doc_id = str(uuid4())
            dest_name = f"{doc_id}{src.suffix.lower()}"
            dest_path = self.docs_dir / dest_name
            shutil.copy2(src, dest_path)

            doc = {
                "id": doc_id,
                "filename": dest_name,
                "original_name": src.name,
                "path": str(dest_path),
                "title": "",
                "authors": "",
                "year": "",
                "venue": "",
                "url": "",
                "summary": ""
            }
            self.library.append(doc)
            self.save_library()
            self.refresh_doc_list()
        except Exception as e:
            messagebox.showerror(tr("status_error"), tr("msg_pdf_add_failed").format(str(e)))

    def remove_selected(self):
        tr = self.app.tr
        if not self.selected_doc_id:
            messagebox.showwarning(tr("status_error"), tr("msg_select_doc"))
            return

        doc = self.get_doc(self.selected_doc_id)
        if not doc:
            return

        try:
            path = Path(doc.get("path", ""))
            if path.exists():
                path.unlink()
            self.library = [d for d in self.library if d["id"] != self.selected_doc_id]
            self.save_library()
            self.selected_doc_id = None
            self.refresh_doc_list()
            self.clear_meta_fields()
            self.library_status.configure(text=tr("msg_doc_removed"))
        except Exception as e:
            messagebox.showerror(tr("status_error"), tr("msg_pdf_add_failed").format(str(e)))

    def clear_meta_fields(self):
        self.title_var.set("")
        self.authors_var.set("")
        self.year_var.set("")
        self.venue_var.set("")
        self.url_var.set("")
        self.summary_box.delete("0.0", "end")
        self.citations_box.delete("0.0", "end")

    def save_metadata(self):
        if not self.selected_doc_id:
            messagebox.showwarning(self.app.tr("status_error"), self.app.tr("msg_select_doc"))
            return

        doc = self.get_doc(self.selected_doc_id)
        if not doc:
            return
        doc["title"] = self.title_var.get().strip()
        doc["authors"] = self.authors_var.get().strip()
        doc["year"] = self.year_var.get().strip()
        doc["venue"] = self.venue_var.get().strip()
        doc["url"] = self.url_var.get().strip()
        self.save_library()
        self.update_citations(doc)

    def generate_summary(self):
        tr = self.app.tr
        if not self.selected_doc_id:
            messagebox.showwarning(tr("status_error"), tr("msg_select_doc"))
            return

        doc = self.get_doc(self.selected_doc_id)
        if not doc:
            return

        path = Path(doc.get("path", ""))
        if not path.exists():
            messagebox.showerror(tr("status_error"), tr("msg_pdf_add_failed").format("PDF not found"))
            return

        text = self.extract_pdf_text(path)
        if not text:
            messagebox.showwarning(tr("status_error"), tr("msg_no_text"))
            return

        summary = self.extractive_summary(text, max_sentences=5)
        doc["summary"] = summary
        self.save_library()
        self.summary_box.delete("0.0", "end")
        self.summary_box.insert("end", summary)

    def update_citations(self, doc):
        apa = self.format_citation(doc, "apa")
        ieee = self.format_citation(doc, "ieee")
        self.citations_box.delete("0.0", "end")
        self.citations_box.insert("end", f"APA:\n{apa}\n\nIEEE:\n{ieee}")

    def copy_citation(self, style):
        tr = self.app.tr
        doc = self.get_doc(self.selected_doc_id) if self.selected_doc_id else None
        if not doc:
            messagebox.showwarning(tr("status_error"), tr("msg_select_doc"))
            return
        citation = self.format_citation(doc, style)
        self.app.clipboard_clear()
        self.app.clipboard_append(citation)
        messagebox.showinfo(tr("btn_copy"), tr("msg_citation_copied"))

    # --- Ask Actions ---
    def save_endpoint(self):
        self.app.set_setting("rag_endpoint", self.endpoint_var.get().strip())

    def ask_question(self):
        tr = self.app.tr
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning(tr("status_error"), tr("msg_missing_question"))
            return

        if not self.index:
            self.ask_status.configure(text=tr("status_no_index"), text_color="orange")
            return

        self.ask_status.configure(text=tr("status_searching"), text_color="gray")
        threading.Thread(target=self._ask_worker, args=(question,), daemon=True).start()

    def _ask_worker(self, question):
        tr = self.app.tr
        try:
            hits = self.search(question, top_k=4)
            if not hits:
                self.after(0, lambda: self.ask_status.configure(text=tr("msg_no_results"), text_color="orange"))
                return

            context_text = "\n\n".join([h["text"] for h in hits])
            sources_text = self.format_sources(hits)

            answer = self.answer_with_llm(question, context_text)
            if not answer:
                answer = context_text

            self.after(0, lambda: self._set_answer(answer, sources_text))
        except Exception as e:
            self.after(0, lambda: self.ask_status.configure(text=tr("msg_answer_failed").format(str(e)), text_color="red"))

    def _set_answer(self, answer, sources_text):
        self.answer_box.delete("0.0", "end")
        self.answer_box.insert("end", answer)
        self.sources_box.delete("0.0", "end")
        self.sources_box.insert("end", sources_text)
        self.ask_status.configure(text=self.app.tr("status_ready"), text_color="gray")

    # --- Indexing ---
    def build_index(self):
        tr = self.app.tr
        if not self.library:
            self.library_status.configure(text=tr("status_no_docs"), text_color="orange")
            return

        if not PyPDF2:
            messagebox.showerror(tr("status_error"), tr("msg_pdf_dependency"))
            return

        self.library_status.configure(text=tr("status_index_building"), text_color="gray")
        threading.Thread(target=self._build_index_worker, daemon=True).start()

    def _build_index_worker(self):
        tr = self.app.tr
        chunks = []
        try:
            for doc in self.library:
                path = Path(doc.get("path", ""))
                if not path.exists():
                    continue
                text = self.extract_pdf_text(path)
                if not text:
                    continue
                doc_chunks = self.chunk_text(text)
                for idx, chunk in enumerate(doc_chunks):
                    tf = self.term_freq(self.tokenize(chunk))
                    chunks.append({
                        "doc_id": doc["id"],
                        "chunk_id": idx,
                        "text": chunk,
                        "tf": tf
                    })

            self.index = chunks
            self.save_index()
            self.idf = self.build_idf(self.index)
            self.after(0, lambda: self.library_status.configure(text=tr("msg_index_done"), text_color="green"))
        except Exception as e:
            self.after(0, lambda: self.library_status.configure(text=tr("msg_index_failed").format(str(e)), text_color="red"))

    # --- Core Helpers ---
    def load_library(self):
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def save_library(self):
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.library, f, indent=2)

    def load_index(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False)

    def get_doc(self, doc_id):
        for doc in self.library:
            if doc["id"] == doc_id:
                return doc
        return None

    def extract_pdf_text(self, path: Path) -> str:
        if not PyPDF2:
            return ""
        try:
            reader = PyPDF2.PdfReader(str(path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(parts)
        except Exception:
            return ""

    def tokenize(self, text: str):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    def term_freq(self, tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def build_idf(self, chunks):
        df = {}
        for chunk in chunks:
            seen = set(chunk.get("tf", {}).keys())
            for t in seen:
                df[t] = df.get(t, 0) + 1
        n = max(len(chunks), 1)
        idf = {}
        for t, count in df.items():
            idf[t] = math.log((n + 1) / (count + 1)) + 1.0
        return idf

    def tfidf(self, tf):
        vec = {}
        for t, count in tf.items():
            idf = self.idf.get(t)
            if idf:
                vec[t] = count * idf
        return vec

    def cosine(self, v1, v2):
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        for t, w in v1.items():
            if t in v2:
                dot += w * v2[t]
        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query, top_k=4):
        q_tf = self.term_freq(self.tokenize(query))
        q_vec = self.tfidf(q_tf)
        scored = []
        for chunk in self.index:
            c_vec = self.tfidf(chunk.get("tf", {}))
            score = self.cosine(q_vec, c_vec)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def chunk_text(self, text, chunk_size=700, overlap=120):
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(tokens):
                break
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(tokens):
                break
        return chunks

    def split_sentences(self, text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def extractive_summary(self, text, max_sentences=5):
        sentences = self.split_sentences(text)
        if not sentences:
            return ""
        sentence_tfs = [self.term_freq(self.tokenize(s)) for s in sentences]
        df = {}
        for tf in sentence_tfs:
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1
        n = len(sentence_tfs)
        idf = {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
        scores = []
        for i, tf in enumerate(sentence_tfs):
            score = sum((idf.get(t, 0.0) * c) for t, c in tf.items())
            scores.append((score, i, sentences[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scores[:max_sentences], key=lambda x: x[1])
        return " ".join([s for _, _, s in top])

    def format_authors(self, authors_raw, style):
        if not authors_raw:
            return ""
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        if style == "apa":
            return ", ".join(authors)
        if style == "ieee":
            return ", ".join(authors)
        return ", ".join(authors)

    def format_citation(self, doc, style):
        title = doc.get("title") or doc.get("original_name") or "Documento"
        authors = self.format_authors(doc.get("authors", ""), style)
        year = doc.get("year") or "s.f."
        venue = doc.get("venue") or ""
        url = doc.get("url") or ""

        if style == "apa":
            parts = [authors, f"({year}).", title]
            if venue:
                parts.append(venue)
            if url:
                parts.append(url)
            return " ".join(p for p in parts if p).strip()

        if style == "ieee":
            parts = [authors, f"\"{title},\""]
            if venue:
                parts.append(venue + ",")
            parts.append(year + ".")
            if url:
                parts.append(url)
            return " ".join(p for p in parts if p).strip()

        return title

    def format_sources(self, hits):
        lines = []
        for i, hit in enumerate(hits, start=1):
            doc = self.get_doc(hit["doc_id"])
            if doc:
                title = doc.get("title") or doc.get("original_name") or doc.get("filename")
            else:
                title = "Documento"
            lines.append(f"[{i}] {title}")
        return "\n".join(lines)

    def answer_with_llm(self, question, context):
        endpoint = self.endpoint_var.get().strip()
        if not endpoint:
            return ""

        system_prompt = (
            "Responde usando solo el contexto provisto. "
            "Si no hay suficiente informacion, explica la limitacion."
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta:\n{question}"}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        try:
            r = requests.post(endpoint, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception:
            return ""
