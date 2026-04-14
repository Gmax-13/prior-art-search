import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Outputs from './pages/Outputs';
import Insights from './pages/Insights';

function App() {
  return (
    <Router>
      <Navbar />
      <main className="page-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/outputs" element={<Outputs />} />
          <Route path="/insights" element={<Insights />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
