import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // In GitHub Actions build for https://imvynor.github.io/Qorin/, use '/Qorin/'
  // Locally or on custom domain, use '/'
  base: process.env.VITE_BASE_PATH || (process.env.GITHUB_ACTIONS ? '/Qorin/' : '/'),
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
