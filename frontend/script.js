// ============================================================
// MED-RAG FRONTEND
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";


// ============================================================
// GLOBAL STATE
// ============================================================

let currentFile = null;
let currentCitations = [];


// ============================================================
// UPLOAD PDF
// ============================================================

async function uploadPDF(input) {

    if (!input.files || input.files.length === 0) {
        return;
    }

    const file = input.files[0];

    currentFile = file;

    // Show document/chat areas
    document.getElementById("emptyChat").style.display = "none";
    document.getElementById("documentArea").style.display = "flex";
    document.getElementById("chatArea").style.display = "block";

    // Show filename
    const documentTitle =
        document.querySelector(".document-card h3");

    if (documentTitle) {
        documentTitle.innerText = file.name;
    }

    // Clear previous chat
    const messages =
        document.getElementById("messages");

    if (messages) {
        messages.innerHTML = "";
    }

    resetSourcePreview();

    // --------------------------------------------------------
    // Upload to backend
    // --------------------------------------------------------

    const formData = new FormData();

    formData.append("file", file);

    try {

        const response = await fetch(
            `${API_BASE_URL}/upload-pdf`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to process the PDF."
            );
        }

        console.log(
            "PDF uploaded successfully:",
            data
        );

        alert(
            `PDF uploaded successfully!\n\n` +
            `Pages: ${data.pages}\n` +
            `Chunks: ${data.chunks}\n` +
            `Stored: ${data.stored_in_qdrant}`
        );

    } catch (error) {

        console.error(
            "UPLOAD ERROR:",
            error
        );

        alert(
            "Unable to process the PDF.\n\n" +
            error.message
        );
    }
}


// ============================================================
// START WITH PDF
// ============================================================

async function startWithPDF(input) {

    await uploadPDF(input);
}


// ============================================================
// SEND QUESTION
// ============================================================

async function sendQuestion() {

    const questionInput =
        document.getElementById("question");

    const question =
        questionInput.value.trim();

    if (!question) {

        alert(
            "Please enter a question."
        );

        return;
    }

    const messages =
        document.getElementById("messages");

    // --------------------------------------------------------
    // USER MESSAGE
    // --------------------------------------------------------

    const userMessage =
        document.createElement("div");

    userMessage.className =
        "user-message";

    userMessage.innerHTML = `
        <div class="message">
            ${escapeHTML(question)}
        </div>
    `;

    messages.appendChild(
        userMessage
    );

    // Clear question box
    questionInput.value = "";

    // --------------------------------------------------------
    // Loading message
    // --------------------------------------------------------

    const aiMessage =
        document.createElement("div");

    aiMessage.className =
        "ai-message";

    aiMessage.innerHTML = `
        <div class="ai-icon">
            ✚
        </div>

        <div class="answer">

            <p>
                Searching your medical document...
            </p>

        </div>
    `;

    messages.appendChild(
        aiMessage
    );

    aiMessage.scrollIntoView({
        behavior: "smooth"
    });

    // --------------------------------------------------------
    // ASK BACKEND
    // --------------------------------------------------------

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/ask`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })
                }
            );

        const data =
            await response.json();

        console.log(
            "RAG RESPONSE:",
            data
        );

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to get answer."
            );
        }

        // ----------------------------------------------------
        // Save citations
        // ----------------------------------------------------

        currentCitations =
            data.citations || [];

        console.log(
            "CITATIONS:",
            currentCitations
        );

        // ----------------------------------------------------
        // Display answer
        // ----------------------------------------------------

        displayAnswer(
            aiMessage,
            data
        );

    } catch (error) {

        console.error(
            "ASK ERROR:",
            error
        );

        const answer =
            aiMessage.querySelector(
                ".answer"
            );

        answer.innerHTML = `
            <p>
                Unable to connect to the MED-RAG backend.
            </p>

            <p style="font-size: 13px;">
                Please make sure FastAPI is running on
                ${API_BASE_URL}
            </p>
        `;
    }
}


// ============================================================
// DISPLAY ANSWER
// ============================================================

function displayAnswer(
    aiMessage,
    data
) {

    const answer =
        aiMessage.querySelector(
            ".answer"
        );

    const answerText =
        data.answer ||
        "No answer was generated.";

    // --------------------------------------------------------
    // Build citation buttons
    // --------------------------------------------------------

    let citationHTML = "";

    if (
        data.citations &&
        data.citations.length > 0
    ) {

        citationHTML = `
            <div class="sources">

                <span>
                    Sources:
                </span>

                <div class="citation-list">
        `;

        data.citations.forEach(
            (citation, index) => {

                const page =
                    citation.page;

                const filename =
                    citation.filename;

                citationHTML += `
                    <button
                        class="citation-button"
                        onclick="showCitation(${index})"
                    >
                        📄 Page ${page}
                    </button>
                `;
            }
        );

        citationHTML += `
                </div>

            </div>
        `;

    } else {

        citationHTML = `
            <div class="sources">
                <span>
                    Sources: No citation available
                </span>
            </div>
        `;
    }

    // --------------------------------------------------------
    // Update answer
    // --------------------------------------------------------

    answer.innerHTML = `

        <p>
            ${escapeHTML(answerText)}
        </p>

        ${citationHTML}

    `;

    // --------------------------------------------------------
    // Automatically show first citation
    // --------------------------------------------------------

    if (
        data.citations &&
        data.citations.length > 0
    ) {

        showCitation(0);
    }
}


// ============================================================
// SHOW CITATION
// ============================================================

function showCitation(index) {

    if (
        !currentCitations ||
        !currentCitations[index]
    ) {

        console.warn(
            "Citation not found:",
            index
        );

        return;
    }

    const citation =
        currentCitations[index];

    console.log(
        "Showing citation:",
        citation
    );

    // --------------------------------------------------------
    // Page
    // --------------------------------------------------------

    const pageElement =
        document.querySelector(".page");

    if (pageElement) {

        pageElement.innerText =
            `Page ${citation.page}`;
    }

    // --------------------------------------------------------
    // Source preview
    // --------------------------------------------------------

    const preview =
        document.querySelector(
            ".source-preview p"
        );

    if (preview) {

        preview.innerHTML = `
            <strong>
                ${escapeHTML(
            citation.filename
        )}
            </strong>

            <br><br>

            ${escapeHTML(
            citation.text
        )}
        `;
    }

    // --------------------------------------------------------
    // Scroll to source preview
    // --------------------------------------------------------

    const sourcePreview =
        document.getElementById(
            "sourcePreview"
        );

    if (sourcePreview) {

        sourcePreview.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }
}


// ============================================================
// SHOW SOURCE
// ============================================================

function showSource(page) {

    if (
        !currentCitations ||
        currentCitations.length === 0
    ) {

        return;
    }

    const index =
        currentCitations.findIndex(
            citation =>
                `Page ${citation.page}` === page
        );

    if (index !== -1) {

        showCitation(index);
    }
}


// ============================================================
// RESET SOURCE PREVIEW
// ============================================================

function resetSourcePreview() {

    const pageElement =
        document.querySelector(".page");

    if (pageElement) {

        pageElement.innerText =
            "Page --";
    }

    const preview =
        document.querySelector(
            ".source-preview p"
        );

    if (preview) {

        preview.innerHTML =
            "Ask a question to see relevant information from the uploaded document.";
    }

    currentCitations = [];
}


// ============================================================
// CLEAR CHAT
// ============================================================

function clearChat() {

    const messages =
        document.getElementById(
            "messages"
        );

    if (messages) {

        messages.innerHTML = "";
    }

    const question =
        document.getElementById(
            "question"
        );

    if (question) {

        question.value = "";
    }

    resetSourcePreview();
}


// ============================================================
// NEW CHAT
// ============================================================

function newChat() {

    document.getElementById(
        "documentArea"
    ).style.display = "none";

    document.getElementById(
        "chatArea"
    ).style.display = "none";

    document.getElementById(
        "emptyChat"
    ).style.display = "flex";

    const question =
        document.getElementById(
            "question"
        );

    if (question) {

        question.value = "";
    }

    const title =
        document.querySelector(
            ".document-card h3"
        );

    if (title) {

        title.innerText =
            "No document uploaded";
    }

    const messages =
        document.getElementById(
            "messages"
        );

    if (messages) {

        messages.innerHTML = "";
    }

    resetSourcePreview();

    currentFile = null;
}


// ============================================================
// ATTACH PDF
// ============================================================

function attachPDF() {

    const fileInput =
        document.querySelector(
            'input[type="file"]'
        );

    if (fileInput) {

        fileInput.click();

    } else {

        alert(
            "PDF upload input not found."
        );
    }
}


// ============================================================
// THEME
// ============================================================

function changeTheme() {

    document.body.classList.toggle(
        "dark"
    );
}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHTML(value) {

    if (value === null ||
        value === undefined) {

        return "";
    }

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}