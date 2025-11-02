import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Landing from './components/Landing'
import Session from './components/Session'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/session/:id" element={<Session />} />
      </Routes>
    </Router>
  )
}

export default App  