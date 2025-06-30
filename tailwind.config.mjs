/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}",
    "./node_modules/preline/preline.js",
  ],
  theme: {
    fontFamily: {
      sans: [
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Roboto",
        "Helvetica",
        "Arial",
        "sans-serif",
      ],
      serif: [
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Georgia",
        "Cambria",
        "Times New Roman",
        "serif",
      ],
      mono: [
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Menlo",
        "Monaco",
        "Consolas",
        "Liberation Mono",
        "Courier New",
        "monospace",
      ],
    },
    extend: {
      backgroundImage: {
        "gradient-to-t":
          "linear-gradient(to top, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 0) 100%)",
        transparent:
          "linear-gradient(to top, rgba(0, 0, 0, 0) 0%, rgba(0, 0, 0, 0) 100%)",
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            a: {
              color: theme("colors.blue.500"),
              textDecoration: "none",
              "&:hover": {
                color: theme("colors.blue.700"),
                textDecoration: "underline",
              },
            },
            sup: {
              fontSize: "inherit",
              verticalAlign: "baseline !important",
              position: "static",
              color: theme("colors.blue.500"),
              "&::before": { content: '"["' },
              "&::after": { content: '"]"' },
              "&:hover": {
                textDecoration: "underline",
                color: theme("colors.blue.700"),
              },
            },
            "a.footnote-ref": {
              "&::before": { content: '"["' },
              "&::after": { content: '"]"' },
            },
          },
        },
        dark: {
          css: {
            a: {
              color: theme("colors.blue.400"),
              textDecoration: "none",
              "&:hover": {
                color: theme("colors.blue.600"),
                textDecoration: "underline",
              },
            },
            sup: {
              fontSize: "inherit",
              verticalAlign: "baseline !important",
              position: "static",
              color: theme("colors.blue.400"),
              "&::before": { content: '"["' },
              "&::after": { content: '"]"' },
              "&:hover": {
                textDecoration: "underline",
                color: theme("colors.blue.600"),
              },
            },
            "a.footnote-ref": {
              "&::before": { content: '"["' },
              "&::after": { content: '"]"' },
            },
          },
        },
      }),
    },
  },
  plugins: [require("preline/plugin"), require("@tailwindcss/typography")],
};
