function LoadingDots() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px]">
      <div className="flex gap-1 mb-4">
        <div className="w-2 h-2 bg-loading rounded-full animate-pulse" style={{ animationDelay: '0ms', animationDuration: '1.5s' }}></div>
        <div className="w-2 h-2 bg-loading rounded-full animate-pulse" style={{ animationDelay: '150ms', animationDuration: '1.5s' }}></div>
        <div className="w-2 h-2 bg-loading rounded-full animate-pulse" style={{ animationDelay: '300ms', animationDuration: '1.5s' }}></div>
      </div>
      <p className="text-text-primary text-lg">Generating your personalized learning experience...</p>
    </div>
  )
}

export default LoadingDots