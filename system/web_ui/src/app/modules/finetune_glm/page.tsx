"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type FinetuneState = {
  deps: Record<string, string>;
  deps_ok: boolean;
  running: boolean;
  output_dir: string;
  script_ok: boolean;
};

export default function FinetuneGLMPage() {
  const { translations } = useApp();
  const [state, setState] = useState<FinetuneState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [datasetPath, setDatasetPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [epochs, setEpochs] = useState("1.0");
  const [batchSize, setBatchSize] = useState("1");
  const [gradAccum, setGradAccum] = useState("8");
  const [learningRate, setLearningRate] = useState("0.0002");
  const [maxSeq, setMaxSeq] = useState("4096");
  const [loraR, setLoraR] = useState("16");
  const [loraAlpha, setLoraAlpha] = useState("16");
  const [loraDropout, setLoraDropout] = useState("0.0");
  const [exportGguf, setExportGguf] = useState(true);
  const [ggufQuant, setGgufQuant] = useState("q4_k_m");

  const refresh = async () => {
    const data = await fetchJson<FinetuneState>("/modules/finetune_glm/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/finetune_glm/logs");
    setLogs(data.lines);
  };

  useEffect(() => {
    refresh();
    refreshLogs();
    const id = setInterval(() => {
      refresh();
      refreshLogs();
    }, 5000);
    return () => clearInterval(id);
  }, []);

  const installDeps = async () => {
    await fetchJson("/modules/finetune_glm/deps", { method: "POST" });
  };

  const runFinetune = async () => {
    await fetchJson("/modules/finetune_glm/run", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: datasetPath,
        output_dir: outputDir || null,
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        grad_accum: Number(gradAccum),
        learning_rate: Number(learningRate),
        max_seq_len: Number(maxSeq),
        lora_r: Number(loraR),
        lora_alpha: Number(loraAlpha),
        lora_dropout: Number(loraDropout),
        export_gguf: exportGguf,
        gguf_quant: ggufQuant,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/finetune_glm/open_output", { method: "POST" });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.finetune_title || "GLM-4.7 Fine-tune"}</div>
        <h1>{translations.finetune_title || "GLM-4.7 Fine-tune"}</h1>
        <p>{translations.finetune_subtitle || "Assistant to fine-tune GLM-4.7-Flash locally with Unsloth."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.finetune_setup_title || "Setup"}</h2>
          <span className="pill">{state?.deps_ok ? "Deps OK" : "Deps missing"}</span>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            <button className="primary" onClick={installDeps}>
              {translations.finetune_btn_install || "Install dependencies"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.finetune_btn_open_output || "Open output"}
            </button>
            <a className="ghost" href="https://unsloth.ai/docs/models/glm-4.7-flash" target="_blank">
              {translations.finetune_btn_open_docs || "Open guide"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.finetune_dataset_title || "Dataset"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.finetune_dataset_label || "Dataset path"}
              <input
                value={datasetPath}
                onChange={(event) => setDatasetPath(event.target.value)}
                placeholder={translations.finetune_dataset_placeholder || "dataset.jsonl"}
              />
            </label>
            <label>
              {translations.finetune_base_model || "Base model"}
              <input value="unsloth/GLM-4.7-Flash" readOnly />
            </label>
            <label>
              {translations.finetune_format_label || "Format"}
              <input
                value={translations.finetune_format_sharegpt || "ShareGPT JSONL (messages or conversations)"}
                readOnly
              />
            </label>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.finetune_train_title || "Training"}</h2>
        </div>
        <div className="panel-body">
          <div className="grid-two">
            <label>
              {translations.finetune_epochs || "Epochs"}
              <input value={epochs} onChange={(event) => setEpochs(event.target.value)} />
            </label>
            <label>
              {translations.finetune_batch_size || "Batch size"}
              <input value={batchSize} onChange={(event) => setBatchSize(event.target.value)} />
            </label>
            <label>
              {translations.finetune_grad_accum || "Grad accumulation"}
              <input value={gradAccum} onChange={(event) => setGradAccum(event.target.value)} />
            </label>
            <label>
              {translations.finetune_learning_rate || "Learning rate"}
              <input value={learningRate} onChange={(event) => setLearningRate(event.target.value)} />
            </label>
            <label>
              {translations.finetune_max_seq || "Max sequence length"}
              <input value={maxSeq} onChange={(event) => setMaxSeq(event.target.value)} />
            </label>
            <label>
              {translations.finetune_lora_r || "LoRA r"}
              <input value={loraR} onChange={(event) => setLoraR(event.target.value)} />
            </label>
            <label>
              {translations.finetune_lora_alpha || "LoRA alpha"}
              <input value={loraAlpha} onChange={(event) => setLoraAlpha(event.target.value)} />
            </label>
            <label>
              {translations.finetune_lora_dropout || "LoRA dropout"}
              <input value={loraDropout} onChange={(event) => setLoraDropout(event.target.value)} />
            </label>
          </div>
          <label style={{ marginTop: "1rem" }}>
            {translations.finetune_output_label || "Output folder"}
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
          </label>
          <div className="grid-two" style={{ marginTop: "0.8rem" }}>
            <label>
              {translations.finetune_gguf_label || "Export GGUF"}
              <select
                value={exportGguf ? "yes" : "no"}
                onChange={(event) => setExportGguf(event.target.value === "yes")}
              >
                <option value="yes">yes</option>
                <option value="no">no</option>
              </select>
            </label>
            <label>
              {translations.finetune_gguf_quant_label || "GGUF quantization"}
              <select value={ggufQuant} onChange={(event) => setGgufQuant(event.target.value)}>
                <option value="q4_k_m">q4_k_m</option>
                <option value="q5_k_m">q5_k_m</option>
                <option value="q8_0">q8_0</option>
                <option value="f16">f16</option>
              </select>
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={runFinetune} disabled={!state?.script_ok}>
              {translations.finetune_btn_run || "Run fine-tune"}
            </button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.finetune_log_label || "Log"}</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
