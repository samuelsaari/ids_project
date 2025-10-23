const input = document.getElementById('username-input');
const dropdown = document.getElementById('dropdown');
const alertBox = document.getElementById('no-match-alert');
let usernames = [];

fetch('/usernames')
  .then(res => res.json())
  .then(data => usernames = data);

input.addEventListener('input', function() {
  const query = this.value
  const lowercase_query = query.toLowerCase();
  dropdown.innerHTML = '';

  if (!query) {
    dropdown.style.display = 'none';
    alertBox.style.display = 'none';
    return;
  }

  const filtered = usernames
    .filter(u => u.toLowerCase().startsWith(lowercase_query))
    .slice(0, 10);

    console.log(query)
    console.log(usernames.includes(query))
  if (usernames.includes(query)) {
    alertBox.style.display = 'none';
  } else {
    alertBox.style.display = 'block';
  }

  if (filtered.length === 0) {
    dropdown.style.display = 'none';
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
      alertBox.style.display = 'none';
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
