"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";
import { fetchJson } from "@/lib/webBridge";

type Service = {
  key: string;
  name: string;
  description: string;
  port: number;
  installed: boolean;
  running: boolean;
};

type MonitorData = {
  system: {
    cpu_name: string;
    cpu_threads: number;
    ram_gb: number;
    ram_used_gb?: number;
    ram_available_gb?: number;
    ram_percent?: number;
    cpu_percent?: number;
    cpu_temp_c?: number | null;
    cuda_available: boolean;
    gpu_names: string[];
    vram_gb: number[];
    gpu_util?: number | null;
    gpu_temp_c?: number | null;
    gpu_mem_used_gb?: number | null;
    gpu_mem_total_gb?: number | null;
  };
  services: Service[];
  models: { label: string; path: string; recommended: boolean }[];
};

const rankGpu = (name: string, vram: number) => {
  const lowered = name.toLowerCase();
  let score = vram || 0;
  if (lowered.includes("nvidia")) score += 1000;
  if (lowered.includes("amd") || lowered.includes("radeon")) score += 500;
  if (lowered.includes("intel")) score += 200;
  if (lowered.includes("virtual") || lowered.includes("microsoft") || lowered.includes("basic render")) {
    score -= 1000;
  }
  return score;
};

const pickGpu = (names: string[] = [], vram: number[] = []) => {
  if (!names.length) {
    return { name: "", vram: 0 };
  }
  const pairs = names.map((name, idx) => ({ name, vram: vram[idx] || 0 }));
  pairs.sort((a, b) => rankGpu(b.name, b.vram) - rankGpu(a.name, a.vram));
  return pairs[0];
};

export default function MonitorPage() {
  const { translations } = useApp();
  const [data, setData] = useState<MonitorData | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [modelSelection, setModelSelection] = useState<Record<string, string>>({});

  const refresh = async () => {
    const res = await fetchJson<MonitorData>("/monitor");
    setData(res);
    if (res.models?.length) {
      setModelSelection((prev) => ({
        ...prev,
        llm_service: prev.llm_service || res.models[0].path,
        clm_service: prev.clm_service || res.models[0].path,
      }));
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const trigger = async (key: string, action: "start" | "stop" | "install" | "uninstall") => {
    setBusyKey(key);
    const body: Record<string, string> = { key };
    if (action === "start" && (key === "llm_service" || key === "clm_service")) {
      body.model_path = modelSelection[key];
    }
    await fetchJson(`/services/${action}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    await refresh();
    setBusyKey(null);
  };

  if (!data) {
    return (
      <AppShell>
        <div className="panel">
          <div className="panel-body">Cargando monitor...</div>
        </div>
      </AppShell>
    );
  }

  const gpu = pickGpu(data.system.gpu_names, data.system.vram_gb);

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.monitor_title || "AI Endpoints"}</div>
        <h1>{translations.monitor_title || "AI Endpoints"}</h1>
        <p>Controla los servicios locales y revisa el hardware activo.</p>
      </div>

      <section className="stats">
        <div className="stat-card">
          <div className="stat-label">CPU</div>
          <div className="stat-value">
            {data.system.cpu_name} ({data.system.cpu_threads} Threads)
          </div>
          <div className="list-meta">
            {data.system.cpu_percent !== undefined ? `Uso ${data.system.cpu_percent.toFixed(0)}%` : "Uso N/A"}
            {data.system.cpu_temp_c !== null && data.system.cpu_temp_c !== undefined
              ? ` · Temp ${data.system.cpu_temp_c.toFixed(0)}°C`
              : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">RAM</div>
          <div className="stat-value">{data.system.ram_gb} GB</div>
          <div className="list-meta">
            {data.system.ram_used_gb !== undefined
              ? `Usada ${data.system.ram_used_gb} GB`
              : "Usada N/A"}
            {data.system.ram_available_gb !== undefined
              ? ` · Libre ${data.system.ram_available_gb} GB`
              : ""}
            {data.system.ram_percent !== undefined ? ` · ${data.system.ram_percent.toFixed(0)}%` : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">GPU</div>
          <div className="stat-value">
            {gpu.name ? `${gpu.name} (${gpu.vram || 0} GB)` : "N/A"}
          </div>
          <div className="list-meta">
            {data.system.gpu_util !== null && data.system.gpu_util !== undefined
              ? `Uso ${data.system.gpu_util.toFixed(0)}%`
              : "Uso N/A"}
            {data.system.gpu_temp_c !== null && data.system.gpu_temp_c !== undefined
              ? ` · Temp ${data.system.gpu_temp_c.toFixed(0)}°C`
              : ""}
            {data.system.gpu_mem_used_gb !== null && data.system.gpu_mem_used_gb !== undefined
              ? ` · VRAM ${data.system.gpu_mem_used_gb.toFixed(1)}/${(data.system.gpu_mem_total_gb || 0).toFixed(1)} GB`
              : ""}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.lbl_api_services || "API Services"}</h2>
        </div>
        <div className="panel-body">
          <div className="list">
            {data.services.map((service) => (
              <div key={service.key} className="list-row">
                <div>
                  <div className="list-title">{service.name}</div>
                  <div className="list-meta">{service.description}</div>
                  <div className="list-meta">Port: {service.port}</div>
                </div>
                <div className="list-actions monitor-actions">
                  <span className="pill">
                    {service.running
                      ? translations.svc_running || "Running"
                      : translations.svc_stopped || "Stopped"}
                  </span>
                  {(service.key === "llm_service" || service.key === "clm_service") && (
                    <select
                      value={modelSelection[service.key] || ""}
                      onChange={(event) =>
                        setModelSelection((prev) => ({ ...prev, [service.key]: event.target.value }))
                      }
                    >
                      {data.models.length ? (
                        data.models.map((model) => (
                          <option key={model.path} value={model.path}>
                            {model.label}
                          </option>
                        ))
                      ) : (
                        <option value="">No models found</option>
                      )}
                    </select>
                  )}
                  {!service.installed && (
                    <button
                      className="primary"
                      disabled={busyKey === service.key}
                      onClick={() => trigger(service.key, "install")}
                    >
                      {translations.btn_install || "Install"}
                    </button>
                  )}
                  {service.installed && (
                    <>
                      <button
                        className="ghost"
                        disabled={busyKey === service.key}
                        onClick={() => trigger(service.key, "uninstall")}
                      >
                        {translations.svc_uninstall || "Uninstall"}
                      </button>
                      <button
                        className="primary"
                        disabled={busyKey === service.key}
                        onClick={() => trigger(service.key, service.running ? "stop" : "start")}
                      >
                        {service.running ? translations.svc_stop || "Stop" : translations.svc_start || "Start"}
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </AppShell>
  );
}
