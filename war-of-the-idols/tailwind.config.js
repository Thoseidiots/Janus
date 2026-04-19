/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'void': '#0a0010',
        'ash': '#1a0a00',
        'frost': '#001020',
        'soul': '#6b21a8',
        'ember': '#7c2d12',
      },
      fontFamily: {
        'gothic': ['Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}
