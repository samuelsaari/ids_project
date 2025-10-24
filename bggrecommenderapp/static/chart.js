function initChart(recId, recRatings) {
  const canvas = document.getElementById(recId);
  const ctx = canvas.getContext('2d');

  const labels = Object.keys(recRatings);
  const values = Object.values(recRatings);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(12, 112, 95, 0.7)',
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}
