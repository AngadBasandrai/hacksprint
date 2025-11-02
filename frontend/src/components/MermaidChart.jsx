import { useEffect, useRef, useState } from 'react'
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'
import mermaid from 'mermaid'

const wrapText = (text, limit = 30) => {
  const words = text.split(" ");
  let lines = [];
  let current = "";
  for (let w of words) {
    if ((current + w).length > limit) {
      lines.push(current.trim());
      current = w + " ";
    } else {
      current += w + " ";
    }
  }
  lines.push(current.trim());
  return lines.join("<br>");
};

function MermaidChart({ diagram }) {
  const ref = useRef(null)
  const containerRef = useRef(null)
  const [svg, setSvg] = useState('')
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!diagram) return

    // FIX: convert escaped newlines into real newlines before rendering
    let cleanDiagram = diagram.replace(/\\n/g, '\n').trim()

    // Process text labels in the diagram with wrapText
    // This handles common mermaid label patterns
    cleanDiagram = cleanDiagram.replace(/(["'])(.*?)\1/g, (match, quote, text) => {
      // Only wrap if text is longer than 30 characters
      if (text.length > 30) {
        const wrapped = wrapText(text, 30)
        return `${quote}${wrapped}${quote}`
      }
      return match
    })

    // Also handle label patterns like A["Long Text Here"]
    cleanDiagram = cleanDiagram.replace(/\["([^"]+)"/g, (match, text) => {
      if (text.length > 30) {
        const wrapped = wrapText(text, 30)
        return `["${wrapped}"`
      }
      return match
    })

    // Handle label patterns like A[Long Text Here]
    cleanDiagram = cleanDiagram.replace(/\[([^\]]+)\]/g, (match, text) => {
      if (text.length > 30 && !text.includes('"')) {
        const wrapped = wrapText(text, 30)
        return `[${wrapped}]`
      }
      return match
    })

    const renderMermaid = async () => {
      try {
        mermaid.initialize({
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'loose',
          logLevel: 1, // Reduce console noise
        })

        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`
        const { svg: svgCode } = await mermaid.render(id, cleanDiagram)
        setSvg(svgCode)
      } catch (error) {
        console.warn('Mermaid diagram rendering failed, using fallback:', error.message)
        // Create a user-friendly fallback SVG instead of showing error
        const fallbackSvg = `
          <svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1" rx="8"/>
            <text x="50%" y="40%" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#6c757d">
              Diagram Unavailable
            </text>
            <text x="50%" y="65%" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#adb5bd">
              Please regenerate for a new diagram
            </text>
          </svg>
        `
        setSvg(fallbackSvg)
      }
    }

    renderMermaid()
  }, [diagram])

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.25, 3))
  }

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.25, 0.5))
  }

  const handleReset = () => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }

  const handleWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      const delta = e.deltaY > 0 ? -0.1 : 0.1
      setScale(prev => Math.max(0.5, Math.min(3, prev + delta)))
    }
  }

  const handleMouseDown = (e) => {
    if (e.button === 0) {
      const startX = e.clientX - position.x
      const startY = e.clientY - position.y
      setIsDragging(true)
      setDragStart({ x: startX, y: startY })
    }
  }

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e) => {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, dragStart])

  if (!diagram) return null

  if (svg) {
    return (
      <div className="relative">
        {/* Zoom Controls */}
        <div className="absolute top-2 right-2 z-10 flex gap-2 bg-white/90 backdrop-blur-sm rounded-lg p-2 border border-border shadow-lg">
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded hover:bg-card transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4 text-text-primary" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded hover:bg-card transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4 text-text-primary" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 rounded hover:bg-card transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw className="w-4 h-4 text-text-primary" />
          </button>
        </div>

        {/* Scrollable Diagram Container */}
        <div
          ref={containerRef}
          className="w-full overflow-auto bg-white border border-border rounded-lg"
          style={{ maxHeight: '600px', cursor: isDragging ? 'grabbing' : 'grab' }}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
        >
          <div
            ref={ref}
            className="flex justify-center items-center p-4"
            style={{
              transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
              transformOrigin: 'top left',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out'
            }}
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="w-full overflow-x-auto p-4">
      <div className="flex justify-center items-center min-h-[200px]">
        <div className="text-black">Rendering diagram...</div>
      </div>
    </div>
  )
}

export default MermaidChart