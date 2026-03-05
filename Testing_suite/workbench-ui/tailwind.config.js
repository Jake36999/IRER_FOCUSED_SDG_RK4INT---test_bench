module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'cyber-black': '#0a0a0a',
        'terminal-green': '#22c55e',
        'critical-red': '#ef4444',
        'amber-warning': '#eab308',
      },
      borderColor: {
        'critical-red': '#ef4444',
      },
      textColor: {
        'terminal-green': '#22c55e',
      },
      animation: {
        'pulse-slow': 'pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'backdrop-blur': 'backdrop-blur 0.5s ease-in-out',
      },
    },
  },
  plugins: [],
};
