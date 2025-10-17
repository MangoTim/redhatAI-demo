let pdfScanned = false;

async function loadChatHistory() {
  try {
    const res = await fetch('http://192.168.147.108:5000/history');
    const history = await res.json();
    const container = document.getElementById('chatHistory');
    container.innerHTML = history.map(h => `
      <div class="chat-row">
        <div class="chat-bubble ${h.role}">
          <div class="chat-text">${h.message.replace(/\n/g, '<br>')}</div>
          <div class="chat-timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        </div>
      </div>
    `).join('');
    container.scrollTop = container.scrollHeight;
  } catch (error) {
    console.error("Failed to load chat history:", error);
  }
}

async function sendMessage() {
  const message = document.getElementById('messageInput').value;
  const model = document.getElementById('modelSelector').value;
  if (!message.trim()) return;

  const chatHistory = document.getElementById('chatHistory');

  chatHistory.innerHTML += `
    <div class="chat-row">
      <div class="chat-bubble user">
        <div class="chat-text">${message}</div>
        <div class="chat-timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
      </div>
    </div>
  `;
  chatHistory.scrollTop = chatHistory.scrollHeight;
  document.getElementById('messageInput').value = "";

  const loadingBubble = document.createElement("div");
  loadingBubble.className = "chat-row";
  loadingBubble.id = "loadingBubble";
  loadingBubble.innerHTML = `
    <div class="chat-bubble assistant loading">
      <div class="chat-text">
        Thinking...
        <span class="spinner"></span>
      </div>
      <div class="chat-timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
    </div>
  `;
  chatHistory.appendChild(loadingBubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  try {
    const endpoint = pdfScanned ? '/ask_pdf' : '/chat';
    const res = await fetch(`http://192.168.147.108:5000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, model })
    });

    let data;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Failed to parse JSON:", e);
      const text = await res.text();
      console.error("Raw response:", text);
      loadingBubble.remove();
      return;
    }

    console.log("Received from", endpoint, ":", data);

    const reply = data.response || data.answer || data.error || 'No response';

    loadingBubble.innerHTML = `
      <div class="chat-bubble assistant">
        <div class="chat-text">${reply.replace(/\n/g, '<br>')}</div>
        <div class="chat-timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
      </div>
    `;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  } catch (error) {
    console.error("Error sending message:", error);
    loadingBubble.remove();
  }
}

function toggleFeatureBox() {
  const box = document.getElementById("featureBox");
  box.style.display = box.style.display === "none" ? "block" : "none";
}

async function uploadPDF() {
  const file = document.getElementById("pdfInput").files[0];
  const formData = new FormData();
  formData.append("pdf", file);

  try {
    const res = await fetch('http://192.168.147.108:5000/upload_pdf', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    pdfScanned = true;

    document.getElementById("featureBox").style.display = "none";
    const featureBtn = document.getElementById("featureBtn");
    featureBtn.outerHTML = `<button id="removePdfBtn" onclick="removePDF()">Remove PDF</button>`;
  } catch (error) {
    alert("Upload failed: " + error);
  }
}

async function removePDF() {
  try {
    const res = await fetch('http://192.168.147.108:5000/remove_pdf', {
      method: 'POST'
    });
    const data = await res.json();
    pdfScanned = false;

    const removeBtn = document.getElementById("removePdfBtn");
    removeBtn.outerHTML = `<button id="featureBtn" onclick="toggleFeatureBox()">Features</button>`;
  } catch (error) {
    alert("Remove failed: " + error);
  }
}

window.onload = loadChatHistory;