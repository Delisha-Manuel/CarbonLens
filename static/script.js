const carbonForm = document.getElementById("carbonForm");
if (carbonForm) {
    carbonForm.addEventListener("submit", function(event) {
        event.preventDefault();
        const formData = {
            diet: document.getElementById("diet").value,
            shower: document.getElementById("shower").value,
            heating: document.getElementById("heating").value,
            transport: document.getElementById("transport").value,
            vehicle: document.getElementById("vehicle").value,
            social: document.getElementById("social").value,
            waste_size: document.getElementById("waste_size").value,
            energy: document.getElementById("energy").value,
            distance: Number(document.getElementById("distance").value),
            air: Number(document.getElementById("air").value),
            grocery: Number(document.getElementById("grocery").value),
            waste_count: Number(document.getElementById("waste_count").value),
            clothes: Number(document.getElementById("clothes").value),
            internet: Number(document.getElementById("internet").value),
            tv_pc: Number(document.getElementById("tv_pc").value),
            recycling: Array.from(document.querySelectorAll('input[name="recycling"]:checked')).map(cb => cb.value),
            cooking: Array.from(document.querySelectorAll('input[name="cooking"]:checked')).map(cb => cb.value),
        };
        sessionStorage.setItem("carbonData", JSON.stringify(formData));
        window.location.href = "/results";
    });
};

document.addEventListener("DOMContentLoaded", () => {
    if (!document.getElementById("footprint")) return;

    const data = JSON.parse(sessionStorage.getItem("carbonData"));
    if (!data) {
        document.getElementById("footprint").innerText = "No data found.";
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        const predictedFootprint = result.prediction;
        document.getElementById("footprint").innerHTML =
            `🌍 Estimated Annual Footprint: <strong>${predictedFootprint} kg CO₂</strong>`;

        const adjustedValues = result.feature_values.map(v => v === 0 ? 0.01 : v);

        new Chart(document.getElementById("impactChart"), {
            type: "bar",
            data: {
                labels: result.feature_labels,
                datasets: [{
                    data: adjustedValues,
                    backgroundColor: [
                        "#326e00","#00a505","#a5d6a7","#ffe082","#ff8a65",
                        "#80cbc4","#4dd0e1","#ffd54f","#ff8a65","#ba68c8",
                        "#7986cb","#e57373","#a1887f","#90a4ae"
                    ]
                }]
            },
            options: {
                indexAxis: "y",
                plugins: { legend: { display: false } },
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: { display: true, text: "Predicted Contribution to Carbon Footprint (kg)" },
                    }
                }
            }
        });
        
        const rawResponse = result.advice;

        if (typeof marked !== 'undefined' && rawResponse) {
            const renderedHTML = marked.parse(rawResponse);
            document.getElementById('advice').innerHTML = renderedHTML;
        } else {
            document.getElementById('advice').innerText = rawResponse || "No advice available.";
        }
    });
});