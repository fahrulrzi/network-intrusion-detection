// Dynamic loader for navbar and footer + active link highlighter
async function loadComponents() {
  const navPh = document.getElementById('navbar-placeholder');
  const footPh = document.getElementById('footer-placeholder');

  if (navPh) {
    try {
      const res = await fetch('/components/navbar.html');
      navPh.innerHTML = await res.text();
      highlightActiveNav();
      wireMobileNav();
    } catch (e) { console.error('Navbar load error:', e); }
  }

  if (footPh) {
    try {
      const res = await fetch('/components/footer.html');
      footPh.innerHTML = await res.text();
    } catch (e) { console.error('Footer load error:', e); }
  }
}

function highlightActiveNav() {
  const path = window.location.pathname;
  const links = document.querySelectorAll('nav a[data-route]');
  links.forEach(a => {
    const route = a.getAttribute('data-route');
    if (!route) return;
    if (route === '/' && (path === '/' || path === '')) {
      a.classList.add('text-[#c084fc]', 'font-semibold');
    } else if (route !== '/' && path.startsWith(route)) {
      a.classList.add('text-[#c084fc]', 'font-semibold');
    } else {
      a.classList.remove('text-[#c084fc]', 'font-semibold');
    }
  });
}

function wireMobileNav() {
  const btn = document.getElementById('mobile-menu-btn');
  const menu = document.getElementById('mobile-menu');
  if (btn && menu) {
    btn.addEventListener('click', () => menu.classList.toggle('hidden'));
  }
}

// Initialize
window.addEventListener('DOMContentLoaded', loadComponents);
