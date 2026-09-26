const API = "http://127.0.0.1:8000";

let currentDocuments = [];
let chatHistory = [];
let currentChatId = null;

let lastAnswer = "";
let lastQuestion = "";

const suggestionPool = [
    "What is the patient's age?",
    "What are the symptoms?",
    "What are the vital signs?",
    "List the medications",
    "Are there any allergies?",
    "What is the diagnosis?",
    "Summarize the documents"
];

document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    updateDocuments();

    document
        .getElementById("question")
        .focus();
});


function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
}


function formatAnswer(text) {
    return escapeHTML(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}


/* =========================
   FILE UPLOAD
========================= */

function openFilePicker() {
    document
        .getElementById("fileInput")
        .click();
}


async function handleFiles(fileList) {

    const files = Array.from(fileList || []);

    if (!files.length) return;


    const validFiles = files.filter(file => {

        const allowed = [
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/webp"
        ];

        if (!allowed.includes(file.type)) {

            alert(
                `${file.name} is not a supported file.`
            );

            return false;
        }


        if (file.size > 50 * 1024 * 1024) {

            alert(
                `${file.name} is larger than 50 MB.`
            );

            return false;
        }

        return true;
    });


    if (!validFiles.length) return;


    showUploading(validFiles);


    try {

        for (const file of validFiles) {
            await uploadSingleFile(file);
        }


        updateDocuments();


        document
            .getElementById("welcome")
            .style.display = "none";


        if (!currentChatId) {
            createCurrentChat();
        }


        saveCurrentChat();


        addSystemMessage(
            validFiles.length === 1
                ? `${validFiles[0].name} is ready.`
                : `${validFiles.length} documents are ready.`
        );


        document
            .getElementById("question")
            .focus();


    } catch (error) {

        console.error(error);

        alert(
            error.message ||
            "Document upload failed."
        );

        updateDocuments();
    }
}


async function uploadSingleFile(file) {

    const formData = new FormData();

    formData.append("file", file);


    const response = await fetch(
        `${API}/upload-pdf`,
        {
            method: "POST",
            body: formData
        }
    );


    const data = await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            `Upload failed: ${file.name}`
        );
    }


    const existingIndex =
        currentDocuments.findIndex(
            doc => doc.name === file.name
        );


    const documentData = {

        name: file.name,

        type: file.type,

        size: file.size,

        pages: data.pages || 1,

        ocr: data.ocr_used || false,

        ocrPages: data.ocr_pages || [],

        documentId:
            data.document_id ||
            data.id ||
            "",

        url:
            URL.createObjectURL(file)
    };


    if (existingIndex >= 0) {

        if (
            currentDocuments[existingIndex].url
        ) {

            URL.revokeObjectURL(
                currentDocuments[existingIndex].url
            );
        }


        currentDocuments[existingIndex] =
            documentData;

    } else {

        currentDocuments.push(
            documentData
        );
    }
}


/* =========================
   DOCUMENT LIST
========================= */

function showUploading(files) {

    const box =
        document.getElementById("documents");


    box.innerHTML = files.map(file => `

        <div class="document-item">

            <div class="file-icon">
                ...
            </div>

            <div class="document-name">

                <b>
                    ${escapeHTML(file.name)}
                </b>

                <span>
                    Processing OCR and document text...
                </span>

            </div>

        </div>

    `).join("");
}


function updateDocuments() {

    const box =
        document.getElementById("documents");


    document
        .getElementById("documentCount")
        .textContent =
        currentDocuments.length;


    if (!currentDocuments.length) {

        box.innerHTML = "";

        return;
    }


    box.innerHTML =
        currentDocuments.map(
            (doc, index) => `

            <div class="document-item">

                <div class="file-icon">

                    ${
                        doc.type === "application/pdf"
                            ? "PDF"
                            : "IMG"
                    }

                </div>


                <div class="document-name">

                    <b
                        title="${escapeHTML(doc.name)}">

                        ${escapeHTML(doc.name)}

                    </b>

                    <span>

                        ${doc.pages || 1}
                        page${(doc.pages || 1) === 1 ? "" : "s"}

                        ${
                            doc.ocr
                                ? " • OCR ready"
                                : ""
                        }

                    </span>

                </div>


                <button
                    class="doc-remove"
                    onclick="removeDocument(${index})"
                    title="Remove">

                    ×

                </button>

            </div>

        `
        ).join("");
}


function removeDocument(index) {

    const doc =
        currentDocuments[index];


    if (!doc) return;


    if (doc.url) {

        URL.revokeObjectURL(
            doc.url
        );
    }


    currentDocuments.splice(
        index,
        1
    );


    updateDocuments();


    if (!currentDocuments.length) {

        document
            .getElementById("welcome")
            .style.display = "block";
    }


    saveCurrentChat();
}


/* =========================
   CHAT CREATION
========================= */

function createCurrentChat() {

    if (!currentDocuments.length)
        return;


    currentChatId =
        "chat_" +
        Date.now() +
        "_" +
        Math.random()
            .toString(36)
            .slice(2);


    const chat = {

        id: currentChatId,

        documents:
            currentDocuments.map(
                doc => ({

                    name: doc.name,

                    type: doc.type,

                    size: doc.size,

                    pages: doc.pages,

                    ocr: doc.ocr,

                    documentId:
                        doc.documentId || ""

                })
            ),

        messages: [],

        createdAt:
            new Date().toISOString(),

        updatedAt:
            new Date().toISOString()
    };


    chatHistory.unshift(chat);

    saveHistoryToStorage();

    renderHistory();
}


/* =========================
   SAVE CURRENT CHAT
========================= */

function saveCurrentChat() {

    if (!currentChatId)
        return;


    const chat =
        chatHistory.find(
            item =>
                item.id === currentChatId
        );


    if (!chat) return;


    chat.documents =
        currentDocuments.map(
            doc => ({

                name: doc.name,

                type: doc.type,

                size: doc.size,

                pages: doc.pages,

                ocr: doc.ocr,

                documentId:
                    doc.documentId || ""

            })
        );


    chat.updatedAt =
        new Date().toISOString();


    saveHistoryToStorage();

    renderHistory();
}


/* =========================
   SAVE MESSAGE
========================= */

function saveMessage(
    role,
    text,
    sources = []
) {

    if (!currentChatId)
        return;


    const chat =
        chatHistory.find(
            item =>
                item.id === currentChatId
        );


    if (!chat) return;


    chat.messages.push({

        role: role,

        text: text,

        sources: sources,

        time:
            new Date().toISOString()
    });


    chat.updatedAt =
        new Date().toISOString();


    saveHistoryToStorage();

    renderHistory();
}


/* =========================
   SEND QUESTION
========================= */

async function sendQuestion() {

    const input =
        document.getElementById("question");


    const question =
        input.value.trim();


    if (!question)
        return;


    if (!currentDocuments.length) {

        alert(
            "Please upload at least one medical document first."
        );

        return;
    }


    if (!currentChatId) {
        createCurrentChat();
    }


    lastQuestion =
        question;


    addUserMessage(
        question
    );


    saveMessage(
        "user",
        question
    );


    input.value = "";

    autoResize(input);


    document
        .getElementById("suggestions")
        .innerHTML = "";


    const loading =
        addLoadingMessage();


    scrollChat();


    try {

        const documentNames =
            currentDocuments.map(
                doc => doc.name
            );


        const documentIds =
            currentDocuments
                .map(
                    doc =>
                        doc.documentId
                )
                .filter(Boolean);


        const response =
            await fetch(
                `${API}/ask`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            question:
                                question,

                            documents:
                                documentNames,

                            document_ids:
                                documentIds
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Question failed."
            );
        }


        loading.remove();


        lastAnswer =
            data.answer || "";


        const sources =
            data.sources || [];


        addAIMessage(
            data.answer ||
            "No answer was generated.",
            sources
        );


        saveMessage(
            "assistant",
            data.answer ||
            "No answer was generated.",
            sources
        );


        saveCurrentChat();


    } catch (error) {

        loading.remove();

        console.error(error);


        addAIMessage(
            "I couldn't get an answer right now. Please check that the MED-RAG backend is running.",
            []
        );
    }
}


/* =========================
   USER MESSAGE
========================= */

function addUserMessage(text) {

    const messages =
        document.getElementById(
            "messages"
        );


    const row =
        document.createElement("div");


    row.className =
        "message-row user";


    row.innerHTML = `

        <div class="user-bubble">

            ${escapeHTML(text)}

        </div>

    `;


    messages.appendChild(row);


    scrollToMessage(row);
}


/* =========================
   LOADING
========================= */

function addLoadingMessage() {

    const messages =
        document.getElementById(
            "messages"
        );


    const row =
        document.createElement("div");


    row.className =
        "ai-row";


    row.innerHTML = `

        <div class="ai-avatar">
            ✦
        </div>

        <div class="ai-card">

            <p>
                Reading your documents...
            </p>

        </div>

    `;


    messages.appendChild(row);


    scrollToMessage(row);


    return row;
}


/* =========================
   AI MESSAGE
========================= */

function addAIMessage(
    answer,
    sources
) {

    const messages =
        document.getElementById(
            "messages"
        );


    const row =
        document.createElement("div");


    row.className =
        "ai-row";


    const uniqueSources = [];

    const seen =
        new Set();


    sources.forEach(
        source => {

            const key =
                `${source.filename}|${source.page}`;


            if (!seen.has(key)) {

                seen.add(key);

                uniqueSources.push(
                    source
                );
            }

        }
    );


    const sourceHTML =
        uniqueSources.length
            ? `

                <div class="sources">

                    <span class="sources-label">
                        Sources:
                    </span>

                    ${
                        uniqueSources
                            .map(
                                source => `

                                <button
                                    type="button"
                                    class="source-btn"
                                    onclick='showSource(${JSON.stringify(source)})'>

                                    ${escapeHTML(
                                        source.filename
                                    )}

                                    • Page
                                    ${source.page || 1}

                                </button>

                            `
                            )
                            .join("")
                    }

                </div>

            `
            : "";


    row.innerHTML = `

        <div class="ai-avatar">
            ✦
        </div>

        <div class="ai-card">

            <p>
                ${formatAnswer(answer)}
            </p>

            ${sourceHTML}

        </div>

    `;


    messages.appendChild(row);


    scrollToMessage(row);
}


/* =========================
   CHAT AUTO SCROLL
========================= */

function scrollChat() {

    const chat =
        document.getElementById(
            "chatScroll"
        );


    requestAnimationFrame(() => {

        chat.scrollTo({

            top:
                chat.scrollHeight,

            behavior:
                "smooth"

        });

    });
}


function scrollToMessage(element) {

    requestAnimationFrame(() => {

        element.scrollIntoView({

            behavior:
                "smooth",

            block:
                "start"

        });

    });
}


/* =========================
   RECENT CHAT HISTORY
========================= */

function saveHistoryToStorage() {

    localStorage.setItem(
        "medrag_chat_history",
        JSON.stringify(
            chatHistory
        )
    );
}


function loadHistory() {

    try {

        chatHistory =
            JSON.parse(
                localStorage.getItem(
                    "medrag_chat_history"
                ) || "[]"
            );

    } catch {

        chatHistory = [];

    }


    renderHistory();
}


function renderHistory() {

    const list =
        document.getElementById(
            "chatList"
        );


    if (!chatHistory.length) {

        list.innerHTML = "";

        return;
    }


    const sorted =
        [...chatHistory].sort(
            (a, b) =>
                new Date(b.updatedAt) -
                new Date(a.updatedAt)
        );


    list.innerHTML =
        sorted.map(
            chat => {

                const count =
                    chat.messages
                        ? chat.messages.length
                        : 0;


                const questionCount =
                    Math.floor(
                        count / 2
                    );


                const names =
                    (chat.documents || [])
                        .map(
                            doc =>
                                doc.name
                        );


                let title;


                if (names.length === 1) {

                    title =
                        names[0];

                } else {

                    title =
                        `${names.length} Medical Documents`;
                }


                return `

                    <div
                        class="history-chat ${
                            chat.id === currentChatId
                                ? "active"
                                : ""
                        }">

                        <button
                            type="button"
                            class="history-open"
                            onclick="openChat('${chat.id}')">

                            <div class="history-title">

                                <span class="history-icon">

                                    ${
                                        names.length > 1
                                            ? "📚"
                                            : "📄"
                                    }

                                </span>


                                <span
                                    class="history-name"
                                    title="${escapeHTML(title)}">

                                    ${escapeHTML(title)}

                                </span>

                            </div>


                            <div class="history-meta">

                                ${questionCount}

                                question${
                                    questionCount === 1
                                        ? ""
                                        : "s"
                                }

                            </div>

                        </button>


                        <button
                            type="button"
                            class="history-delete"
                            onclick="deleteChat(event, '${chat.id}')"
                            title="Delete chat">

                            ⋮

                        </button>

                    </div>

                `;

            }
        ).join("");
}


/* =========================
   OPEN OLD CHAT
========================= */

function openChat(chatId) {

    const chat =
        chatHistory.find(
            item =>
                item.id === chatId
        );


    if (!chat)
        return;


    currentChatId =
        chat.id;


    currentDocuments.forEach(
        doc => {

            if (doc.url) {

                URL.revokeObjectURL(
                    doc.url
                );

            }

        }
    );


    currentDocuments = [];


    (chat.documents || [])
        .forEach(
            doc => {

                currentDocuments.push({

                    name:
                        doc.name,

                    type:
                        doc.type,

                    size:
                        doc.size,

                    pages:
                        doc.pages || 1,

                    ocr:
                        doc.ocr || false,

                    url:
                        null,

                    documentId:
                        doc.documentId || ""

                });

            }
        );


    updateDocuments();


    const messages =
        document.getElementById(
            "messages"
        );


    messages.innerHTML = "";


    document
        .getElementById("welcome")
        .style.display = "none";


    (chat.messages || [])
        .forEach(
            message => {

                if (
                    message.role ===
                    "user"
                ) {

                    addUserMessage(
                        message.text
                    );

                } else {

                    addAIMessage(
                        message.text,
                        message.sources || []
                    );

                }

            }
        );


    renderHistory();


    requestAnimationFrame(() => {

        const chatScroll =
            document.getElementById(
                "chatScroll"
            );


        chatScroll.scrollTop =
            chatScroll.scrollHeight;

    });
}


/* =========================
   DELETE ONE CHAT
========================= */

function deleteChat(
    event,
    chatId
) {

    event.stopPropagation();


    const chat =
        chatHistory.find(
            item =>
                item.id === chatId
        );


    if (!chat)
        return;


    const names =
        (chat.documents || [])
            .map(
                doc =>
                    doc.name
            )
            .join(", ");


    const confirmed =
        confirm(
            `Delete this conversation?\n\n${names}`
        );


    if (!confirmed)
        return;


    chatHistory =
        chatHistory.filter(
            item =>
                item.id !== chatId
        );


    saveHistoryToStorage();


    if (currentChatId === chatId) {

        currentChatId =
            null;


        currentDocuments.forEach(
            doc => {

                if (doc.url) {

                    URL.revokeObjectURL(
                        doc.url
                    );

                }

            }
        );


        currentDocuments = [];


        document
            .getElementById(
                "messages"
            )
            .innerHTML = "";


        document
            .getElementById(
                "welcome"
            )
            .style.display = "block";


        updateDocuments();

    }


    renderHistory();
}


/* =========================
   CLEAR ALL HISTORY
========================= */

function clearHistory() {

    if (!chatHistory.length)
        return;


    const confirmed =
        confirm(
            "Delete all recent chat history?"
        );


    if (!confirmed)
        return;


    chatHistory = [];


    localStorage.removeItem(
        "medrag_chat_history"
    );


    renderHistory();
}


/* =========================
   NEW CHAT
========================= */

function newChat() {

    currentDocuments.forEach(
        doc => {

            if (doc.url) {

                URL.revokeObjectURL(
                    doc.url
                );

            }

        }
    );


    currentDocuments = [];

    currentChatId = null;

    lastAnswer = "";

    lastQuestion = "";


    document
        .getElementById(
            "messages"
        )
        .innerHTML = "";


    document
        .getElementById(
            "question"
        )
        .value = "";


    document
        .getElementById(
            "suggestions"
        )
        .innerHTML = "";


    document
        .getElementById(
            "welcome"
        )
        .style.display = "block";


    updateDocuments();

    renderHistory();


    document
        .getElementById(
            "question"
        )
        .focus();
}


/* =========================
   CLEAR CURRENT CHAT MESSAGES
========================= */

function clearCurrentChat() {

    if (!currentChatId)
        return;


    const chat =
        chatHistory.find(
            item =>
                item.id ===
                currentChatId
        );


    if (!chat)
        return;


    chat.messages = [];


    chat.updatedAt =
        new Date().toISOString();


    document
        .getElementById(
            "messages"
        )
        .innerHTML = "";


    document
        .getElementById(
            "question"
        )
        .value = "";


    saveHistoryToStorage();

    renderHistory();
}


/* =========================
   SYSTEM MESSAGE
========================= */

function addSystemMessage(text) {

    const messages =
        document.getElementById(
            "messages"
        );


    const row =
        document.createElement("div");


    row.className =
        "ai-row";


    row.innerHTML = `

        <div class="ai-avatar">
            ✦
        </div>

        <div class="ai-card">

            <p>
                ${escapeHTML(text)}
            </p>

        </div>

    `;


    messages.appendChild(row);


    scrollToMessage(row);
}


/* =========================
   SOURCE PREVIEW
========================= */

function showSource(source) {

    const doc =
        currentDocuments.find(
            item =>
                item.name ===
                source.filename
        );


    document
        .getElementById(
            "sourceTitle"
        )
        .textContent =
        source.filename;


    document
        .getElementById(
            "sourcePage"
        )
        .textContent =
        `Page ${source.page || 1}`;


    document
        .getElementById(
            "sourceText"
        )
        .textContent =
        source.text ||
        "No extracted text available.";


    const preview =
        document.getElementById(
            "preview"
        );


    preview.innerHTML = "";


    if (!doc || !doc.url) {

        preview.innerHTML = `

            <div
                style="
                    padding:40px;
                    text-align:center;
                    color:#7b849d;
                ">

                Upload this document again
                to preview the original page.

            </div>

        `;

    }

    else if (
        doc.type ===
        "application/pdf"
    ) {

        const iframe =
            document.createElement(
                "iframe"
            );


        iframe.src =
            `${doc.url}#page=${
                source.page || 1
            }`;


        iframe.title =
            "Document page preview";


        preview.appendChild(
            iframe
        );

    }

    else {

        const img =
            document.createElement(
                "img"
            );


        img.src =
            doc.url;


        img.alt =
            "Uploaded document";


        preview.appendChild(
            img
        );

    }


    document
        .getElementById(
            "sourceModal"
        )
        .classList.add("open");
}


function closeSource(event) {

    const modal =
        document.getElementById(
            "sourceModal"
        );


    if (
        event &&
        event.target !== modal
    ) {

        return;
    }


    modal.classList.remove(
        "open"
    );
}


/* =========================
   SUGGESTIONS
========================= */

function handleTyping() {

    const input =
        document.getElementById(
            "question"
        );


    const box =
        document.getElementById(
            "suggestions"
        );


    autoResize(input);


    const value =
        input.value
            .trim()
            .toLowerCase();


    if (!value) {

        box.innerHTML = "";

        return;
    }


    const matches =
        suggestionPool
            .filter(
                item =>
                    item
                        .toLowerCase()
                        .includes(value)
            )
            .slice(0, 4);


    box.innerHTML = "";


    matches.forEach(item => {

        const button =
            document.createElement(
                "button"
            );


        button.type = "button";

        button.className =
            "suggestion";


        button.textContent =
            item;


        button.addEventListener(
            "mousedown",
            event => {

                event.preventDefault();

            }
        );


        button.addEventListener(
            "click",
            event => {

                event.preventDefault();

                event.stopPropagation();

                useSuggestion(item);

            }
        );


        box.appendChild(button);

    });
}


function useSuggestion(text) {

    const input =
        document.getElementById(
            "question"
        );


    input.value =
        text;


    autoResize(input);


    document
        .getElementById(
            "suggestions"
        )
        .innerHTML = "";


    input.focus();


    input.setSelectionRange(
        input.value.length,
        input.value.length
    );
}


/* =========================
   ENTER / SHIFT + ENTER
========================= */

function handleEnter(event) {

    if (event.key !== "Enter")
        return;


    // SHIFT + ENTER
    // → create a new line
    if (event.shiftKey) {

        return;
    }


    // ENTER
    // → send question
    event.preventDefault();

    sendQuestion();
}


/* =========================
   TEXTAREA
========================= */

function autoResize(element) {

    element.style.height =
        "auto";


    element.style.height =
        Math.min(
            element.scrollHeight,
            105
        ) + "px";
}


/* =========================
   SHARE
========================= */

function shareChat() {

    const chat =
        chatHistory.find(
            item =>
                item.id ===
                currentChatId
        );


    if (!chat) {
        return;
    }


    const text =
        (chat.messages || [])
            .map(
                message => {

                    const role =
                        message.role ===
                        "user"
                            ? "You"
                            : "MedRAG";


                    return `${role}:\n${message.text}`;

                }
            )
            .join("\n\n");


    if (
        navigator.share
    ) {

        navigator.share({

            title:
                "MedRAG conversation",

            text:
                text

        }).catch(
            () => {}
        );

    }

    else {

        navigator.clipboard
            .writeText(text)
            .then(
                () =>
                    alert(
                        "Conversation copied."
                    )
            );

    }
}


/* =========================
   THEME
========================= */

function changeTheme() {

    document
        .body
        .classList
        .toggle("dark");

}