"use client";

import Link from "next/link";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

export default function Home() {
  const { translations, modules, favorites, refresh } = useApp();
  const installed = modules.filter((mod) => mod.installed);

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

      <section className="panel">
        <div className="panel-header">
          <h2>Modulos instalados</h2>
          <Link className="ghost" href="/modules">
            {translations.btn_open || "Open"}
          </Link>
        </div>
        <div className="panel-body">
          {installed.length === 0 ? (
            <div className="empty">No hay modulos instalados aun.</div>
          ) : (
            <div className="list">
              {installed.map((module) => (
                <div key={module.key} className="list-row">
                  <div>
                    <div className="list-title">
                      {favorites.includes(module.key) ? `* ${module.title}` : module.title}
                    </div>
                    <div className="list-meta">{module.description}</div>
                  </div>
                  <div className="list-actions">
                    <button
                      className="ghost"
                      onClick={async () => {
                        const isFav = favorites.includes(module.key);
                        if (!isFav && favorites.length >= 3) {
                          return;
                        }
                        await fetchJson(isFav ? "/favorites/remove" : "/favorites/add", {
                          method: "POST",
                          body: JSON.stringify({ key: module.key }),
                        });
                        await refresh();
                      }}
                    >
                      {favorites.includes(module.key)
                        ? translations.fav_remove || "Quitar de favoritos"
                        : translations.fav_add || "Agregar a favoritos"}
                    </button>
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
