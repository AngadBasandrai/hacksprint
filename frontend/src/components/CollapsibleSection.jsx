import { useState } from 'react'

function CollapsibleSection({ title, children, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-700 hover:text-icon-on-button"
      >
        <h2 className="text-xl font-semibold text-text-primary">
          {title}
        </h2>
        <svg
          className={`w-5 h-5 text-icon ${
            isOpen ? 'rotate-180' : ''
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>
      {isOpen && (
        <div className="px-6 py-4 border-t border-border">
          {children}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection