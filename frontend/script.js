// PDF UPLOAD

function uploadPDF(input) {

    if (input.files.length === 0) {
        return;
    }

    let file = input.files[0];

    document.getElementById("emptyChat").style.display = "none";

    document.getElementById("documentArea").style.display = "flex";

    document.getElementById("chatArea").style.display = "block";

    document.querySelector(".document-card h3").innerText =
        file.name;

    // Clear old conversation
    document.getElementById("messages").innerHTML = "";

    // Reset source preview
    document.querySelector(".page").innerText = "Page --";

    document.querySelector(".source-preview p").innerHTML =
        "Ask a question to see relevant information from " +
        file.name + ".";

}


// NEW CHAT

function newChat() {

    // Hide old document and chat
    document.getElementById("documentArea").style.display = "none";
    document.getElementById("chatArea").style.display = "none";

    // Show empty screen
    document.getElementById("emptyChat").style.display = "flex";

    // Clear question box
    document.getElementById("question").value = "";

    // Reset document name
    document.querySelector(".document-card h3").innerText =
        "No document uploaded";

    // Clear previous chat messages
    document.querySelector(".user-message .message").innerText = "";

    document.querySelector(".answer").innerHTML =
        "<p>Your document-based answer will appear here.</p>";

    // Reset source preview
    document.querySelector(".page").innerText = "Page --";

    document.querySelector(".source-preview p").innerHTML =
        "Upload a medical PDF to see relevant sources here.";

}

function startWithPDF(input) {

    if (input.files.length === 0) {
        return;
    }

    let file = input.files[0];

    // Hide empty screen
    document.getElementById("emptyChat").style.display = "none";

    // Show document and chat
    document.getElementById("documentArea").style.display = "flex";
    document.getElementById("chatArea").style.display = "block";

    // Show uploaded PDF name
    document.querySelector(".document-card h3").innerText =
        file.name;

    // Clear any previous messages
    document.getElementById("messages").innerHTML = "";

    // Reset source
    document.querySelector(".page").innerText = "Page --";

    document.querySelector(".source-preview p").innerHTML =
        "Ask a question to see relevant information from " +
        file.name + ".";

}


// CLEAR CHAT

function clearChat() {

    document.getElementById("messages").innerHTML = "";

    document.getElementById("question").value = "";

    document.querySelector(".page").innerText =
        "Page --";

    document.querySelector(".source-preview p").innerHTML =
        "Ask a question to see relevant information from your document.";

}


// SEND QUESTION

function sendQuestion() {

    let question =
        document.getElementById("question").value.trim();

    if (question === "") {

        alert("Please enter a question.");

        return;
    }


    // Get message container
    let messages =
        document.getElementById("messages");


    // Create user message

    let userMessage =
        document.createElement("div");

    userMessage.className =
        "user-message";

    userMessage.innerHTML = `
        <div class="message">
            ${question}
        </div>
    `;

    messages.appendChild(userMessage);


    // Create temporary AI response

    let aiMessage =
        document.createElement("div");

    aiMessage.className =
        "ai-message";

    aiMessage.innerHTML = `
        <div class="ai-icon">
            ✚
        </div>

        <div class="answer">

            <p>
                Your document-based answer will
                appear here after connecting
                the RAG backend.
            </p>

            <div class="sources">

                <span>Sources:</span>

                <button onclick="showSource('Page 1')">
                    Page 1
                </button>

            </div>

        </div>
    `;

    messages.appendChild(aiMessage);


    // Clear input
    document.getElementById("question").value = "";


    // Scroll to latest message
    aiMessage.scrollIntoView({
        behavior: "smooth"
    });

}


// ATTACH PDF

function attachPDF() {

    alert(
        "PDF attachment will be connected to the backend later."
    );

}


// SHOW SOURCE

function showSource(page) {

    document.querySelector(".page").innerText = page;

    document.getElementById("sourcePreview")
        .scrollIntoView({
            behavior: "smooth"
        });

}


// THEME

function changeTheme() {
    document.body.classList.toggle("dark");
}

