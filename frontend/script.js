// ============================================================
// MED-RAG FRONTEND
// ============================================================
const API_BASE_URL = "http://127.0.0.1:8000";



// ============================================================
// GLOBAL STATE
// ============================================================

let currentFile = null;
let currentCitations = [];

// Keep all PDFs selected during this chat
let selectedPDFs = [];
let selectedImageURLs = new Map();

// ============================================================
// UPLOAD PDF / MULTIPLE PDFs
// ============================================================
async function uploadPDF(input) {

    if (!input.files || input.files.length === 0) {
        return;
    }

    // Get newly selected PDFs
    const newFiles = Array.from(input.files);

    // Keep selected PDFs and images available for this chat
    selectedPDFs = selectedPDFs.concat(newFiles);

    // Keep local preview URLs for selected images
    newFiles.forEach(file => {
        if (file.type.startsWith("image/")) {
            const key = `${file.name}|${file.size}|${file.lastModified}`;
            if (!selectedImageURLs.has(key)) {
                selectedImageURLs.set(key, URL.createObjectURL(file));
            }
        }
    });

    // Remove accidental duplicate file selections
    selectedPDFs = selectedPDFs.filter(
        (file, index, self) =>
            index === self.findIndex(
                f =>
                    f.name === file.name &&
                    f.size === file.size &&
                    f.lastModified === file.lastModified
            )
    );

    console.log(
        "ALL SELECTED PDFs:",
        selectedPDFs.map(file => file.name)
    );

    // Show document/chat areas
    document.getElementById("emptyChat").style.display = "none";
    document.getElementById("documentArea").style.display = "flex";
    document.getElementById("chatArea").style.display = "block";

    // --------------------------------------------------------
    // Show total selected PDFs
    // --------------------------------------------------------

    const documentTitle =
        document.querySelector(".document-card h3");

    if (documentTitle) {

        if (selectedPDFs.length === 1) {

            documentTitle.innerText =
                selectedPDFs[0].name;

        } else {

            documentTitle.innerText =
                `${selectedPDFs.length} PDFs selected`;
        }
    }

    // --------------------------------------------------------
    // Clear previous chat
    // --------------------------------------------------------

    const messages =
        document.getElementById("messages");

    if (messages) {
        messages.innerHTML = "";
    }

    resetSourcePreview();

    // --------------------------------------------------------
    // Upload ONLY the newly selected PDFs
    // --------------------------------------------------------
    // This prevents uploading the first 5 PDFs again.

    const formData = new FormData();

    newFiles.forEach(function (file) {

        formData.append(
            "files",
            file
        );

    });

    // --------------------------------------------------------
    // Send request
    // --------------------------------------------------------

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/upload-pdf`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        console.log(
            "PDF BATCH RESPONSE:",
            data
        );

        // ----------------------------------------------------
        // Handle backend error
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to process the PDFs."
            );
        }

        // ----------------------------------------------------
        // Display upload result
        // ----------------------------------------------------

        alert(
            `PDF upload completed!\n\n` +
            `This batch: ${data.total_files}\n` +
            `Successful: ${data.successful}\n` +
            `Failed: ${data.failed}\n\n` +
            `Total PDFs selected in this chat: ${selectedPDFs.length}`
        );

        console.log(
            "Successfully uploaded PDFs:",
            data.results
        );

    } catch (error) {

        console.error(
            "UPLOAD ERROR:",
            error
        );

        alert(
            "Unable to process the PDFs.\n\n" +
            error.message
        );
    }

    // Clear the input so the same file picker can be used again
    input.value = "";
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


    // --------------------------------------------------------
    // AI LOADING MESSAGE
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
                Searching your medical documents...
            </p>

        </div>
    `;


    messages.appendChild(
        aiMessage
    );


    questionInput.value = "";


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
        // SAVE CITATIONS
        // ----------------------------------------------------

        currentCitations =
            data.citations || [];


        console.log(
            "CITATIONS:",
            currentCitations
        );


        // ----------------------------------------------------
        // DISPLAY ANSWER
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
    // BUILD CITATIONS
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
            function (citation, index) {

                citationHTML += `
                    <button
                        class="citation-button"
                        onclick="showCitation(${index})"
                    >
                        📄 Page ${escapeHTML(
                            citation.page
                        )}
                    </button>
                `;
            }
        );


        citationHTML += `
                </div>

            </div>
        `;


    } else {

        citationHTML = "";

        // Clear old source preview when there is no citation
        resetSourcePreview();
    }


    // --------------------------------------------------------
    // DISPLAY ANSWER
    // --------------------------------------------------------
    console.log("NEW DISPLAY ANSWER CODE");
    
    answer.innerHTML = `

        <p>
            ${escapeHTML(answerText)}
        </p>

        ${citationHTML}

    `;


    // --------------------------------------------------------
    // SHOW FIRST SOURCE
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
        console.warn("Citation not found:", index);
        return;
    }

    const citation = currentCitations[index];

    const imagePreview = document.getElementById("sourceImagePreview");
    const imageCitationTextPreview = document.querySelector(".source-preview p");
    const citationFilename = (citation.filename || "").toLowerCase();
    const imageExtension = /\\.(png|jpe?g|webp)$/i;
    const imageKey = [...selectedImageURLs.keys()].find(key =>
        key.split("|")[0].toLowerCase() === citationFilename
    );

    if (imagePreview) {
        if (imageExtension.test(citationFilename) && imageKey) {
            imagePreview.src = selectedImageURLs.get(imageKey);
            imagePreview.style.display = "block";
        } else {
            imagePreview.style.display = "none";
            imagePreview.removeAttribute("src");
        }
    }


    // --------------------------------------------------------
    // PAGE
    // --------------------------------------------------------

    const pageElement =
        document.querySelector(".page");

    if (pageElement) {
        pageElement.innerText =
            `Page ${citation.page}`;
    }

    // --------------------------------------------------------
    // SOURCE PREVIEW
    // Show only a short preview
    // --------------------------------------------------------

    const preview =
        document.querySelector(".source-preview p");

    if (preview) {

        let sourceText =
            citation.text || "";

        sourceText =
            sourceText.replace(/\s+/g, " ").trim();

        if (sourceText.length > 500) {

            sourceText =
                sourceText.substring(0, 500).trim();

            sourceText += "...";
        }

        preview.innerHTML = `
            <strong>
                ${escapeHTML(citation.filename)}
            </strong>

            <br><br>

            ${escapeHTML(sourceText)}
        `;
    }

    // --------------------------------------------------------
    // SCROLL TO SOURCE
    // --------------------------------------------------------

    const sourcePreview =
        document.getElementById("sourcePreview");

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


    const pageNumber =
        parseInt(
            String(page).replace(
                "Page ",
                ""
            )
        );


    const index =
        currentCitations.findIndex(
            function (citation) {

                return citation.page === pageNumber;

            }
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
            "Ask a question to see relevant information from the uploaded documents.";
    }


    const imagePreview = document.getElementById("sourceImagePreview");
    if (imagePreview) {
        imagePreview.style.display = "none";
        imagePreview.removeAttribute("src");
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
    selectedPDFs = [];
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

    if (
        value === null ||
        value === undefined
    ) {

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
