/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: "rgb(var(--surface))",
        surface2: "rgb(var(--surface-2))",
        ink: "rgb(var(--ink))",
        muted: "rgb(var(--muted))",
        accent: "rgb(var(--accent))",
        accent2: "rgb(var(--accent-2))",
        panel: "rgb(var(--panel))",
        ring: "rgb(var(--ring))",
      },
    },
  },
  plugins: [],
};
