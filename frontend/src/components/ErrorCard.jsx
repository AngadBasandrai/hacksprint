function ErrorCard({ error, onRetry }) {
  return (
    <div className="border border-border p-8 max-w-2xl mx-auto bg-card rounded-2xl">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-text-primary mb-2">
          Something went wrong
        </h2>
        <p className="text-text-primary mb-6">{error}</p>
        <button
          onClick={onRetry}
          className="border border-border text-text-primary font-semibold py-3 px-6 hover:bg-button hover:text-icon-on-button transition-colors rounded-lg"
        >
          Try Again
        </button>
      </div>
    </div>
  )
}

export default ErrorCard