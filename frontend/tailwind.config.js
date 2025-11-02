/** @type {import('tailwindcss').Config} */
export default {
    darkMode: ["class"],
    content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
  	extend: {
  		colors: {
  			'text-primary': 'var(--color-text-primary)',
  			'text-secondary': 'var(--color-text-secondary)',
  			'text-tertiary': 'var(--color-text-tertiary)',
  			'bg-card': 'var(--color-bg-card)',
  			'bg-card-opacity': 'var(--color-bg-card-opacity)',
  			'bg-sidebar': 'var(--color-bg-sidebar)',
  			'bg-sidebar-opacity': 'var(--color-bg-sidebar-opacity)',
  			'bg-gradient-start': 'var(--color-bg-gradient-start)',
  			'bg-gradient-mid': 'var(--color-bg-gradient-mid)',
  			'bg-gradient-end': 'var(--color-bg-gradient-end)',
  			button: 'var(--color-button)',
  			'button-hover': 'var(--color-button-hover)',
  			'button-opacity': 'var(--color-button-opacity)',
  			border: 'hsl(var(--border))',
  			'border-light': 'var(--color-border-light)',
  			loading: 'var(--color-loading)',
  			'loading-light': 'var(--color-loading-light)',
  			'code-bg': 'var(--color-code-bg)',
  			icon: 'var(--color-icon)',
  			'icon-light': 'var(--color-icon-light)',
  			'icon-on-button': 'var(--color-icon-on-button)',
  			'icon-muted': 'var(--color-icon-muted)',
  			'gray-50': 'var(--color-gray-50)',
  			'gray-100': 'var(--color-gray-100)',
  			'gray-200': 'var(--color-gray-200)',
  			'gray-300': 'var(--color-gray-300)',
  			'gray-400': 'var(--color-gray-400)',
  			'gray-500': 'var(--color-gray-500)',
  			'gray-600': 'var(--color-gray-600)',
  			'gray-700': 'var(--color-gray-700)',
  			'gray-800': 'var(--color-gray-800)',
  			'gray-900': 'var(--color-gray-900)',
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			}
  		},
  		animation: {
  			'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
}