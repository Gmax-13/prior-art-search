import { NavLink } from 'react-router-dom';
import { HiOutlineHome, HiOutlineTableCells, HiOutlineChartBar } from 'react-icons/hi2';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner container">
        <div className="navbar-brand">
          <span className="brand-icon">◆</span>
          <span className="brand-text">Prior Art Search</span>
        </div>
        <ul className="navbar-links">
          <li>
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
              <HiOutlineHome className="nav-icon" />
              <span>Home</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/outputs" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <HiOutlineTableCells className="nav-icon" />
              <span>Outputs</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/insights" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <HiOutlineChartBar className="nav-icon" />
              <span>Insights</span>
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}
