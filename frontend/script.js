document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("audioFile");
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  const output = document.getElementById("output");
  output.innerHTML = "⏳ Processing...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      body: formData
    });

    const result = await response.json();

    if (result.error) {
      output.innerHTML = `<span style="color:red;">❌ Error: ${result.error}</span>`;
    } else {
      output.innerHTML = `
        <b>Language:</b> ${result.language} <br>
        <b>Spoofed:</b> ${result.spoofed ? "🟥 Yes" : "🟩 No"}
      `;
    }
  } catch (err) {
    output.innerHTML = `<span style="color:red;">❌ Request failed. Is backend running?</span>`;
  }
});

