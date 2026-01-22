"use client";

import Link from "next/link";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";
import { useEffect, useState } from "react";

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

export default function Home() {
  const { translations, modules, favorites, refresh } = useApp();
  const installed = modules.filter((mod) => mod.installed);
  const [monitor, setMonitor] = useState<MonitorData | null>(null);

  useEffect(() => {
    let active = true;
    const refreshMonitor = async () => {
      try {
        const data = await fetchJson<MonitorData>("/monitor");
        if (active) {
          setMonitor(data);
        }
      } catch {
        if (active) {
          setMonitor(null);
        }
      }
    };

    refreshMonitor();
    const id = setInterval(refreshMonitor, 5000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  return (
    <AppShell>
      <section className="hero">
        <div>
          <div className="eyebrow">{translations.hero_badge || "AI POWERED STUDIO"}</div>
          <h1>{translations.hero_title || "Create, Analyze, Innovate"}</h1>
          <p>
            {translations.hero_subtitle ||
              "The comprehensive AI platform for UNLZ. Run language, vision, and audio models locally."}
          </p>
        </div>
        <div className="hero-card">
          <div className="hero-title">{translations.sidebar_installed || "Installed"}</div>
          <div className="hero-value">{installed.length} modulos activos</div>
          <div className="hero-meta">
            {modules.length} disponibles en la tienda de modulos.
          </div>
          <div className="hero-meta">
            <Link className="ghost" href="/modules">
              {translations.btn_explore || "Explore Modules"}
            </Link>
          </div>
        </div>
      </section>

      <section className="stats">
        <div className="stat-card">
          <div className="stat-label">{translations.feat_perf_title || "High Performance"}</div>
          <div className="stat-value">GPU/CPU</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">{translations.feat_sec_title || "Secure Environment"}</div>
          <div className="stat-value">Local</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">{translations.feat_mod_title || "Modular Design"}</div>
          <div className="stat-value">{modules.length}</div>
        </div>
      </section>

      <section className="stats">
        <div className="stat-card">
          <div className="stat-label">CPU</div>
          <div className="stat-value">
            {monitor?.system?.cpu_name
              ? `${monitor.system.cpu_name} (${monitor.system.cpu_threads} Threads)`
              : "N/A"}
          </div>
          <div className="list-meta">
            {monitor?.system?.cpu_percent !== undefined ? `Uso ${monitor.system.cpu_percent.toFixed(0)}%` : "Uso N/A"}
            {monitor?.system?.cpu_temp_c !== null && monitor?.system?.cpu_temp_c !== undefined
              ? ` · Temp ${monitor.system.cpu_temp_c.toFixed(0)}°C`
              : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">RAM</div>
          <div className="stat-value">{monitor?.system?.ram_gb ? `${monitor.system.ram_gb} GB` : "N/A"}</div>
          <div className="list-meta">
            {monitor?.system?.ram_used_gb !== undefined ? `Usada ${monitor.system.ram_used_gb} GB` : "Usada N/A"}
            {monitor?.system?.ram_available_gb !== undefined
              ? ` · Libre ${monitor.system.ram_available_gb} GB`
              : ""}
            {monitor?.system?.ram_percent !== undefined ? ` · ${monitor.system.ram_percent.toFixed(0)}%` : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">GPU</div>
          <div className="stat-value">
            {monitor?.system?.gpu_names?.length
              ? (() => {
                  const gpu = pickGpu(monitor.system.gpu_names, monitor.system.vram_gb);
                  return `${gpu.name} (${gpu.vram || 0} GB)`;
                })()
              : "N/A"}
          </div>
          <div className="list-meta">
            {monitor?.system?.gpu_util !== null && monitor?.system?.gpu_util !== undefined
              ? `Uso ${monitor.system.gpu_util.toFixed(0)}%`
              : "Uso N/A"}
            {monitor?.system?.gpu_temp_c !== null && monitor?.system?.gpu_temp_c !== undefined
              ? ` · Temp ${monitor.system.gpu_temp_c.toFixed(0)}°C`
              : ""}
            {monitor?.system?.gpu_mem_used_gb !== null && monitor?.system?.gpu_mem_used_gb !== undefined
              ? ` · VRAM ${monitor.system.gpu_mem_used_gb.toFixed(1)}/${(monitor.system.gpu_mem_total_gb || 0).toFixed(1)} GB`
              : ""}
          </div>
        </div>
      </section>

    </AppShell>
  );
}
