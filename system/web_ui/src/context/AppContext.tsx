"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { fetchJson } from "@/lib/webBridge";

export type ModuleMeta = {
  key: string;
  title: string;
  description: string;
  category: string;
  installed: boolean;
};

export type AppSettings = {
  language: string;
  model_dir?: string;
  show_logs?: boolean;
  theme?: string;
};

type BootstrapData = {
  settings: AppSettings;
  language: string;
  translations: Record<string, string>;
  modules: ModuleMeta[];
  favorites: string[];
};

type AppState = {
  loading: boolean;
  error: string | null;
  language: string;
  settings: AppSettings;
  translations: Record<string, string>;
  modules: ModuleMeta[];
  favorites: string[];
  refresh: () => Promise<void>;
  updateSettings: (updates: Partial<AppSettings>) => Promise<void>;
};

const AppContext = createContext<AppState | null>(null);

const emptyState: AppState = {
  loading: true,
  error: null,
  language: "es",
  settings: { language: "es" },
  translations: {},
  modules: [],
  favorites: [],
  refresh: async () => {},
  updateSettings: async () => {},
};

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<AppSettings>({ language: "es" });
  const [language, setLanguage] = useState("es");
  const [translations, setTranslations] = useState<Record<string, string>>({});
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const [favorites, setFavorites] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchJson<BootstrapData>("/bootstrap");
      setSettings(data.settings);
      setLanguage(data.language);
      setTranslations(data.translations || {});
      setModules(data.modules || []);
      setFavorites(data.favorites || []);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reach Web Bridge";
      setError(message);
    }
    setLoading(false);
  }, []);

  const updateSettings = useCallback(async (updates: Partial<AppSettings>) => {
    try {
      const res = await fetchJson<{ settings: AppSettings }>("/settings", {
        method: "POST",
        body: JSON.stringify(updates),
      });
      setSettings(res.settings);
      setLanguage(res.settings.language || "es");
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update settings";
      setError(message);
    }
  }, [refresh]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const value = useMemo<AppState>(
    () => ({ loading, error, language, settings, translations, modules, favorites, refresh, updateSettings }),
    [loading, error, language, settings, translations, modules, favorites, refresh, updateSettings]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  return useContext(AppContext) || emptyState;
}
