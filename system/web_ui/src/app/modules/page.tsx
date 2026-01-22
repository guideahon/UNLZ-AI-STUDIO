"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";
import { fetchJson } from "@/lib/webBridge";

const CATEGORY_LABELS: Record<string, string> = {
  core: "Core",
  vision: "Vision",
  audio: "Audio",
  motion: "Motion",
  knowledge: "Knowledge",
  tools: "Tools",
};

export default function ModulesPage() {
  const { modules, translations, refresh, favorites } = useApp();
  const [filter, setFilter] = useState<"all" | "installed">("all");
  const [busyKey, setBusyKey] = useState<string | null>(null);

  const filtered = useMemo(() => {
    return filter === "installed" ? modules.filter((m) => m.installed) : modules;
  }, [modules, filter]);

  const onInstall = async (key: string, installed: boolean) => {
    setBusyKey(key);
    await fetchJson(installed ? "/modules/uninstall" : "/modules/install", {
      method: "POST",
      body: JSON.stringify({ key }),
    });
    await refresh();
    setBusyKey(null);
  };

  const onToggleFavorite = async (key: string) => {
    const isFav = favorites.includes(key);
    if (!isFav && favorites.length >= 3) {
      return;
    }
    setBusyKey(key);
    await fetchJson(isFav ? "/favorites/remove" : "/favorites/add", {
      method: "POST",
      body: JSON.stringify({ key }),
    });
    await refresh();
    setBusyKey(null);
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.store_title || "Module Store"}</div>
        <h1>Catalogo de modulos</h1>
        <p>Gestiona los modulos disponibles y su estado de instalacion.</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Filtros</h2>
          <div className="list-actions">
            <button
              className={filter === "all" ? "primary" : "ghost"}
              onClick={() => setFilter("all")}
            >
              Todos
            </button>
            <button
              className={filter === "installed" ? "primary" : "ghost"}
              onClick={() => setFilter("installed")}
            >
              Instalados
            </button>
          </div>
        </div>
        <div className="panel-body">
          {filtered.length === 0 ? (
            <div className="empty">No hay modulos para mostrar.</div>
          ) : (
            <div className="list">
              {filtered.map((module) => (
                <div key={module.key} className="list-row">
                  <div>
                    <div className="list-title">
                      {favorites.includes(module.key) ? `* ${module.title}` : module.title}
                    </div>
                    <div className="list-meta">{module.description}</div>
                  </div>
                  <div className="list-actions">
                    <span className="pill">
                      {CATEGORY_LABELS[module.category] || module.category}
                    </span>
                    {busyKey === module.key && <span className="pill">En progreso</span>}
                    {module.installed && (
                      <button
                        className="ghost"
                        onClick={() => onToggleFavorite(module.key)}
                        disabled={busyKey === module.key || (!favorites.includes(module.key) && favorites.length >= 3)}
                      >
                        {favorites.includes(module.key)
                          ? translations.fav_remove || "Quitar de favoritos"
                          : translations.fav_add || "Agregar a favoritos"}
                      </button>
                    )}
                    <button
                      onClick={() => onInstall(module.key, module.installed)}
                      className={module.installed ? "ghost" : "primary"}
                      disabled={busyKey === module.key}
                    >
                      {module.installed
                        ? translations.svc_uninstall || "Uninstall"
                        : translations.btn_install || "Install"}
                    </button>
                    <Link className="ghost" href={`/docs/${module.key}`}>
                      {translations.btn_docs || "Documentacion"}
                    </Link>
                    <Link className="ghost" href={`/modules/${module.key}`}>
                      {translations.btn_open || "Open"}
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </AppShell>
  );
}
