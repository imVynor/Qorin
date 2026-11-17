import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const isPages = mode === 'pages' || process.env.GITHUB_ACTIONS || process.env.VITE_BASE_PATH;
  return {
    plugins: [react()],
    // In GitHub Pages build for https://imvynor.github.io/Qorin/, use '/Qorin/'
    // Locally or on custom domain, use '/'
    base: process.env.VITE_BASE_PATH || (isPages ? '/Qorin/' : '/'),
    server: {
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});

