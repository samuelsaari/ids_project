const input = document.getElementById('username-input');
const dropdown = document.getElementById('dropdown');
const alertBox = document.getElementById('no-match-alert');
let usernames = [];

fetch('/usernames')
  .then(res => res.json())
  .then(data => usernames = data);

input.addEventListener('input', function() {
  const query = this.value.toLowerCase();
  dropdown.innerHTML = '';

  if (!query) {
    dropdown.style.display = 'none';
    alertBox.style.display = 'none';
    return;
  }

  const filtered = usernames
    .filter(u => u.toLowerCase().startsWith(query))
    .slice(0, 10);

  if (filtered.length === 0) {
    dropdown.style.display = 'none';
    alertBox.style.display = 'block';
    return;
  }

  filtered.forEach(name => {
    const option = document.createElement('button');
    option.type = 'button';
    option.className = 'list-group-item list-group-item-action text-start'
    option.textContent = name;

    option.onclick = () => {
      input.value = name;
      dropdown.style.display = 'none';
    };
    dropdown.appendChild(option);
  });

  dropdown.style.display = 'block';
});

// Hide dropdown if clicked outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.autocomplete-container')) {
    dropdown.style.display = 'none';
  }
});